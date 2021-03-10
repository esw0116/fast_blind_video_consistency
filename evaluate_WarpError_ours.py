#!/usr/bin/python
from __future__ import print_function

### python lib
import os, sys, argparse, glob, re, math, pickle, cv2
from datetime import datetime
import numpy as np

### torch lib
import torch
import torch.nn as nn

### custom lib
from networks.resample2d_package.resample2d import Resample2d
import networks
import utils


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fast Blind Video Temporal Consistency')
    

    ### testing options
    parser.add_argument('-model',           type=str,     default="FlowNet2",   help='Flow model name')
    parser.add_argument('-test_dir',        type=str,     required=True,            help='test model name')
    parser.add_argument('-dataset',         type=str,     required=True,            help='test datasets')
    parser.add_argument('-phase',           type=str,     default="test",           choices=["train", "test"])
    parser.add_argument('-data_dir',        type=str,     default='optical_flow',   help='path to data folder')
    parser.add_argument('-save_dir',        type=str,     default='saved_imgs',     help='path to save folder')
    parser.add_argument('-seq_len',         type=int,     help='sequence length')
    parser.add_argument('-redo',            action="store_true",                    help='redo evaluation')

    opts = parser.parse_args()

    opts.cuda = True
    opts.grads = {} # dict to collect activation gradients (for training debug purpose)

    ### FlowNet options
    opts.rgb_max = 1.0
    opts.fp16 = False

    print(opts)

    if opts.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without -cuda")
    
    ### initialize FlowNet
    print('===> Initializing model from %s...' %opts.model)
    model = networks.__dict__[opts.model](opts)

    if opts.model != 'PWCNet':
        ### load pre-trained FlowNet
        model_filename = os.path.join("pretrained_models", "%s_checkpoint.pth.tar" %opts.model)
        print("===> Load %s" %model_filename)
        checkpoint = torch.load(model_filename)
        model.load_state_dict(checkpoint['state_dict'])

    device = torch.device("cuda" if opts.cuda else "cpu")
    model = model.to(device)
    model.eval()

    method = os.path.basename(os.path.dirname(opts.test_dir))
    output_dir = os.path.join(opts.save_dir, opts.model, method, opts.phase)
    print("Output Dir: ", output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ## print average if result already exists

    metric_filename = os.path.join(output_dir, "WarpError.txt")
    epe_filename = os.path.join(output_dir, "EndPointError.txt")

    if os.path.exists(metric_filename) and not opts.redo:
        print("Output %s exists, skip..." %metric_filename)

        cmd = 'tail -n1 %s' %metric_filename
        utils.run_cmd(cmd)
        sys.exit()
    

    ## flow warping layer
    device = torch.device("cuda" if opts.cuda else "cpu")
    flow_warping = Resample2d().to(device)

    ### load video list

    input_dir = os.path.join(opts.test_dir, opts.dataset)
    video_list = sorted(glob.glob(os.path.join(opts.test_dir, '*')))

    ### start evaluation
    warp_err_all = np.zeros(len(video_list))
    endpoint_err_all = np.zeros(len(video_list))

    for v in range(len(video_list)):
        video = os.path.basename(video_list[v])

        frame_dir = os.path.join(opts.test_dir, video)
        occ_dir = os.path.join(opts.data_dir, opts.model, opts.dataset, opts.phase, "fw_occlusion", video)
        flow_dir = os.path.join(opts.data_dir, opts.model, opts.dataset, opts.phase, "fw_flow", video)
        
        frame_list = sorted(glob.glob(os.path.join(video_list[v], "*.png")))

        warp_err = 0
        endpoint_err = 0
        count = 0
        for t in range(len(frame_list) - 1):
            if t % opts.seq_len == (opts.seq_len - 1):
                continue
            ### load input images
            img1 = utils.read_img(frame_list[t])
            img2 = utils.read_img(frame_list[t+1])

            fname = os.path.splitext(os.path.basename(frame_list[t]))[0]

            ### load flow
            filename = os.path.join(flow_dir, "{}.flo".format(fname))
            flow = utils.read_flo(filename)

            ### load occlusion mask
            filename = os.path.join(occ_dir, "{}.png".format(fname))
            occ_mask = utils.read_img(filename)
            noc_mask = 1 - occ_mask

            with torch.no_grad():
                ## convert to tensor
                img2 = utils.img2tensor(img2).to(device)
                flow = utils.img2tensor(flow).to(device)

                ## warp img2
                warp_img2 = flow_warping(img2, flow)

                ## convert to numpy array
                warp_img2 = utils.tensor2img(warp_img2)


            ## compute warping error
            diff = np.multiply(warp_img2 - img1, noc_mask)
            
            N = np.sum(noc_mask)
            if N == 0:
                N = diff.shape[0] * diff.shape[1] * diff.shape[2]

            current_we = np.sum(np.square(diff)) / N
            warp_err += current_we


            ## compute end-point error
            with torch.no_grad():
                ### convert to tensor
                img1 = utils.img2tensor(img1).to(device)
        
                ### compute fw flow
                fw_flow = model(img1, img2)
                fw_flow = utils.tensor2img(fw_flow)
                flow = utils.tensor2img(flow)

            fw_flow = utils.resize_flow(fw_flow, W_out = flow.shape[1], H_out = flow.shape[0])
            diff_flow = np.sqrt(np.sum(np.square(flow - fw_flow), axis=2))

            current_epe = np.sum(diff_flow) / (diff_flow.shape[0] * diff_flow.shape[1])
            endpoint_err += current_epe

            count += 1

            print("Evaluate Warping Error on {}-{}: video {:02d} / {:02d}, {} \tWE: {:.06f} \tEPE: {:.6f}".format(opts.dataset, opts.phase, v + 1, len(video_list), fname, current_we, current_epe), end='\r')


        warp_err_all[v] = warp_err / count
        endpoint_err_all[v] = endpoint_err / count

    print("\nAverage Warping Error = %f\n" %(warp_err_all.mean()))
    print("\nAverage End-Point Error = %f\n" %(endpoint_err_all.mean()))

    warp_err_all = np.append(warp_err_all, warp_err_all.mean())
    print("Save %s" %metric_filename)
    np.savetxt(metric_filename, warp_err_all, fmt="%f")
    np.savetxt(epe_filename, endpoint_err_all, fmt="%f")
