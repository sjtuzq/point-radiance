
import argparse


def config_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--splatting_r", type=float, default=0.015, help='the radius for the splatting')
    parser.add_argument("--raster_n", type=int, default=15, help='the point number for soft raterization')
    parser.add_argument("--refine_n", type=int, default=2, help='the point number for soft raterization')
    parser.add_argument("--data_r", type=float, default=0.012, help='the point number for soft raterization')
    parser.add_argument("--step", type=str, default='brdf', help='the running step for the algorithm')
    parser.add_argument("--savemodel", type=str, default=None, help='whether to save the model weight or not')

    parser.add_argument("--lr1", type=float, default=22e-3, help='the learning rate for the rgb texture')
    parser.add_argument("--lr2", type=float, default=8e-4, help='the learning rate for the point position')
    parser.add_argument("--lrexp", type=float, default=0.93, help='the coefficient for the exponential lr decay')
    parser.add_argument("--lr_s", type=float, default=0.03, help='the coefficient for the total variance loss')
    parser.add_argument("--img_s", type=int, default=400, help='the coefficient for the total variance loss')
    parser.add_argument("--memitem", type=object, default=None, help='the coefficient for the total variance loss')


    parser.add_argument("--expname", type=str, default='pcdata',
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='../logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='/tigress/qz9238/workspace/workspace/data/nerf/nerf_synthetic/',
                        help='input data directory')
    parser.add_argument("--dataname", type=str, default='hotdog',
                        help='dataset name')
    parser.add_argument("--grey", type=int, default=1,
                        help='whether to set grey or not')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=500,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", default=True,
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=128,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", default=True,
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=500,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='blender',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=1,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", default=True,
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", default=True, help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=500,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=5000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=10000,
                        help='frequency of render_poses video saving')

    # name_list = ['lego','materials','mic','ficus','drums','chair','ship','hotdog']
    # data_r_mapping = {'hotdog':0.012,'lego':0.08,'materials':0.03,'mic':0.15,'ficus':0.15,'drums':0.25,'chair':0.06,'ship':0.03}
    # splatting_mapping = {'hotdog':0.015,'lego':0.010,'materials':0.010,'mic':0.008,'ficus':0.008,'drums':0.008,'chair':0.010,'ship':0.010}

    # args.dataname = name_list[7]
    # dataset = Data(args)
    # args.memitem = dataset.genpc()
    # args.splatting_r = splatting_mapping[args.dataname]
    # args.data_r = data_r_mapping[args.dataname]
    # solve(args)

    # for i in range(8):
    #     args.dataname = name_list[i]
    #     dataset = Data(args)
    #     args.memitem = dataset.genpc()
    #     args.splatting_r = splatting_mapping[args.dataname]
    #     args.data_r = data_r_mapping[args.dataname]
    #     solve(args)

    return parser



