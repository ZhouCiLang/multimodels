# Author: Jacek Komorowski
# Warsaw University of Technology

from models.minkloc import MinkLoc
from models.minkloc_multimodal import MinkLocMultimodal, ResnetFPN, WaveResnetFPN, TestResnetFPN
from misc.utils import MinkLocParams


def model_factory(params: MinkLocParams):
    in_channels = 1

    # MinkLocMultimodal is our baseline MinkLoc++ model producing 256 dimensional descriptor where
    # each modality produces 128 dimensional descriptor
    # MinkLocRGB and MinkLoc3D are single-modality versions producing 256 dimensional descriptor
    ##added by ZRN: wave part
    ##add wave part
    if params.model_params.model == 'WaveMultimodal':
        print('Go WaveMultimodal!\n')
        
        cloud_fe_size = 128
        cloud_fe = MinkLoc(in_channels=1, feature_size=cloud_fe_size, output_dim=cloud_fe_size,
                           planes=[32, 64, 64], layers=[1, 1, 1], num_top_down=1,
                           conv0_kernel_size=5, block='BasicBlock', pooling_method='GeM', net='Wave')
        image_fe_size = 128
        image_fe = WaveResnetFPN(out_channels=image_fe_size, lateral_dim=image_fe_size,
                             fh_num_bottom_up=4, fh_num_top_down=0)
        model = MinkLocMultimodal(cloud_fe, cloud_fe_size, image_fe, image_fe_size, output_dim=cloud_fe_size + image_fe_size)
        
    elif params.model_params.model == 'Wave3D':
        print('Go Wave3D Point Clouds!\n')
        cloud_fe_size = 128
        cloud_fe = MinkLoc(in_channels=1, feature_size=cloud_fe_size, output_dim=cloud_fe_size,
                           planes=[32, 64, 64], layers=[1, 1, 1], num_top_down=1,
                           conv0_kernel_size=5, block='BasicBlock', pooling_method='GeM', net='Wave')
        model = MinkLocMultimodal(cloud_fe, cloud_fe_size, None, 0, output_dim=cloud_fe_size,
                                  dropout_p=None)
    elif params.model_params.model == 'Wave3D256':
        print('Go Wave3D256 Point Clouds!\n')
        cloud_fe_size = 256
        cloud_fe = MinkLoc(in_channels=1, feature_size=cloud_fe_size, output_dim=cloud_fe_size,
                           planes=[32, 64, 64], layers=[1, 1, 1], num_top_down=1,
                           conv0_kernel_size=5, block='BasicBlock', pooling_method='GeM', net='Wave')
        model = MinkLocMultimodal(cloud_fe, cloud_fe_size, None, 0, output_dim=cloud_fe_size,
                                  dropout_p=None)
    elif params.model_params.model == 'WaveRGB':
        print('Go WaveRGB!\n')
        image_fe_size = 128
        image_fe = WaveResnetFPN(out_channels=image_fe_size, lateral_dim=image_fe_size,
                             fh_num_bottom_up=4, fh_num_top_down=0)
        model = MinkLocMultimodal(None, 0, image_fe, image_fe_size, output_dim=image_fe_size)
    elif params.model_params.model == 'WaveRGB256':
        print('Go WaveRGB256!\n')
        image_fe_size = 256
        image_fe = WaveResnetFPN(out_channels=image_fe_size, lateral_dim=image_fe_size,
                             fh_num_bottom_up=4, fh_num_top_down=0)
        model = MinkLocMultimodal(None, 0, image_fe, image_fe_size, output_dim=image_fe_size)        
    elif params.model_params.model == 'WaveRGBnetTest':
        print('Go WaveRGBnetTest!\n')
        ##changed here
        image_fe_size = 256
        image_fe = TestResnetFPN(out_channels=image_fe_size, lateral_dim=image_fe_size,
                             fh_num_bottom_up=4, fh_num_top_down=0)
        model = MinkLocMultimodal(None, 0, image_fe, image_fe_size, output_dim=image_fe_size)
    elif params.model_params.model == 'WaveMultimodalInfer':
        print('Go WaveMultimodalInfer!\n')
        cloud_fe_size = 128
        cloud_fe = MinkLoc(in_channels=1, feature_size=cloud_fe_size, output_dim=cloud_fe_size,
                           planes=[32, 64, 64], layers=[1, 1, 1], num_top_down=1,
                           conv0_kernel_size=5, block='BasicBlock', pooling_method='GeM', net='Wave')
        image_fe_size = 128
        image_fe = WaveResnetFPN(out_channels=image_fe_size, lateral_dim=image_fe_size,
                             fh_num_bottom_up=4, fh_num_top_down=0)
        model = MinkLocMultimodal(cloud_fe, cloud_fe_size, image_fe, image_fe_size, output_dim=cloud_fe_size, fuse_method='infer')
    elif params.model_params.model == 'WaveMultimodalInferAll':
        print('Go WaveMultimodalInferAll!\n')
        cloud_fe_size = 128
        cloud_fe = MinkLoc(in_channels=1, feature_size=cloud_fe_size, output_dim=cloud_fe_size,
                           planes=[32, 64, 64], layers=[1, 1, 1], num_top_down=1,
                           conv0_kernel_size=5, block='BasicBlock', pooling_method='GeM', net='Wave')
        image_fe_size = 128
        image_fe = WaveResnetFPN(out_channels=image_fe_size, lateral_dim=image_fe_size,
                             fh_num_bottom_up=4, fh_num_top_down=0)
        model = MinkLocMultimodal(cloud_fe, cloud_fe_size, image_fe, image_fe_size, output_dim=cloud_fe_size, fuse_method='inferall')
    elif params.model_params.model == 'WaveMultimodalInferAll256':
        print('Go WaveMultimodalInferAll256!\n')
        cloud_fe_size = 256
        cloud_fe = MinkLoc(in_channels=1, feature_size=cloud_fe_size, output_dim=cloud_fe_size,
                           planes=[32, 64, 64], layers=[1, 1, 1], num_top_down=1,
                           conv0_kernel_size=5, block='BasicBlock', pooling_method='GeM', net='Wave')
        image_fe_size = 256
        image_fe = WaveResnetFPN(out_channels=image_fe_size, lateral_dim=image_fe_size,
                             fh_num_bottom_up=4, fh_num_top_down=0)
        model = MinkLocMultimodal(cloud_fe, cloud_fe_size, image_fe, image_fe_size, output_dim=cloud_fe_size, fuse_method='inferall')
    elif params.model_params.model == 'SVT':
        print('Go SVT!\n')
        cloud_fe_size = 128
        cloud_fe = MinkLoc(in_channels=1, feature_size=cloud_fe_size, output_dim=cloud_fe_size,
                           planes=[32, 64], layers=[1, 1], num_top_down=1,
                           conv0_kernel_size=5, block='BasicBlock', pooling_method='GeM', net='SVT')
        image_fe_size = 128
        image_fe = WaveResnetFPN(out_channels=image_fe_size, lateral_dim=image_fe_size,
                             fh_num_bottom_up=4, fh_num_top_down=0)
        model = MinkLocMultimodal(cloud_fe, cloud_fe_size, image_fe, image_fe_size, output_dim=cloud_fe_size, fuse_method='infer')
    elif params.model_params.model == 'SVTinfer128':
        print('Go SVTinfer128!\n')
        cloud_fe_size = 128
        cloud_fe = MinkLoc(in_channels=1, feature_size=cloud_fe_size, output_dim=cloud_fe_size,
                            planes=[32, 64], layers=[1, 1], num_top_down=1,
                           conv0_kernel_size=5, block='BasicBlock', pooling_method='GeM', net='SVT')
        image_fe_size = 128
        image_fe = WaveResnetFPN(out_channels=image_fe_size, lateral_dim=image_fe_size,
                             fh_num_bottom_up=4, fh_num_top_down=0)
        model = MinkLocMultimodal(cloud_fe, cloud_fe_size, image_fe, image_fe_size, output_dim=cloud_fe_size, fuse_method='infer')
    elif params.model_params.model == 'SVTinfer256':
        print('Go SVTinfer256!\n')
        cloud_fe_size = 256
        cloud_fe = MinkLoc(in_channels=1, feature_size=cloud_fe_size, output_dim=cloud_fe_size,
                           planes=[32, 64], layers=[1, 1], num_top_down=1,
                           conv0_kernel_size=5, block='BasicBlock', pooling_method='GeM', net='SVT')
        image_fe_size = 256
        image_fe = WaveResnetFPN(out_channels=image_fe_size, lateral_dim=image_fe_size,
                             fh_num_bottom_up=4, fh_num_top_down=0)
        model = MinkLocMultimodal(cloud_fe, cloud_fe_size, image_fe, image_fe_size, output_dim=cloud_fe_size, fuse_method='infer')
    elif params.model_params.model == 'SVTadd256':
        print('Go SVTadd256!\n')
        cloud_fe_size = 256
        cloud_fe = MinkLoc(in_channels=1, feature_size=cloud_fe_size, output_dim=cloud_fe_size,
                           planes=[32, 64], layers=[1, 1], num_top_down=1,
                           conv0_kernel_size=5, block='BasicBlock', pooling_method='GeM', net='SVT')
        image_fe_size = 256
        image_fe = WaveResnetFPN(out_channels=image_fe_size, lateral_dim=image_fe_size,
                             fh_num_bottom_up=4, fh_num_top_down=0)
        model = MinkLocMultimodal(cloud_fe, cloud_fe_size, image_fe, image_fe_size, output_dim=cloud_fe_size, fuse_method='add')
    elif params.model_params.model == 'SVTcon256':
        print('Go SVTcon256!\n')
        cloud_fe_size = 128
        cloud_fe = MinkLoc(in_channels=1, feature_size=cloud_fe_size, output_dim=cloud_fe_size,
                           planes=[32, 64], layers=[1, 1], num_top_down=1,
                           conv0_kernel_size=5, block='BasicBlock', pooling_method='GeM', net='SVT')
        image_fe_size = 128
        image_fe = WaveResnetFPN(out_channels=image_fe_size, lateral_dim=image_fe_size,
                             fh_num_bottom_up=4, fh_num_top_down=0)
        model = MinkLocMultimodal(cloud_fe, cloud_fe_size, image_fe, image_fe_size, output_dim=cloud_fe_size+image_fe_size)
    elif params.model_params.model == 'SVTQI':
        print('Go SVTQI!\n')
        cloud_fe_size = 128
        cloud_fe = MinkLoc(in_channels=1, feature_size=cloud_fe_size, output_dim=cloud_fe_size,
                           planes=[32, 64], layers=[1, 1], num_top_down=1,
                           conv0_kernel_size=5, block='BasicBlock', pooling_method='GeM', net='SVTQI')
        image_fe_size = 128
        image_fe = WaveResnetFPN(out_channels=image_fe_size, lateral_dim=image_fe_size,
                             fh_num_bottom_up=4, fh_num_top_down=0)
        model = MinkLocMultimodal(cloud_fe, cloud_fe_size, image_fe, image_fe_size, output_dim=cloud_fe_size, fuse_method='infer')
    elif params.model_params.model == 'SVTQIinfer128':
        print('Go SVTQIinfer128!\n')
        cloud_fe_size = 128
        cloud_fe = MinkLoc(in_channels=1, feature_size=cloud_fe_size, output_dim=cloud_fe_size,
                            planes=[32, 64], layers=[1, 1], num_top_down=1,
                           conv0_kernel_size=5, block='BasicBlock', pooling_method='GeM', net='SVTQI')
        image_fe_size = 128
        image_fe = WaveResnetFPN(out_channels=image_fe_size, lateral_dim=image_fe_size,
                             fh_num_bottom_up=4, fh_num_top_down=0)
        model = MinkLocMultimodal(cloud_fe, cloud_fe_size, image_fe, image_fe_size, output_dim=cloud_fe_size, fuse_method='infer')
    elif params.model_params.model == 'SVTQIinfer256':
        print('Go SVTQIinfer256!\n')
        cloud_fe_size = 256
        cloud_fe = MinkLoc(in_channels=1, feature_size=cloud_fe_size, output_dim=cloud_fe_size,
                           planes=[32, 64], layers=[1, 1], num_top_down=1,
                           conv0_kernel_size=5, block='BasicBlock', pooling_method='GeM', net='SVTQI')
        image_fe_size = 256
        image_fe = WaveResnetFPN(out_channels=image_fe_size, lateral_dim=image_fe_size,
                             fh_num_bottom_up=4, fh_num_top_down=0)
        model = MinkLocMultimodal(cloud_fe, cloud_fe_size, image_fe, image_fe_size, output_dim=cloud_fe_size, fuse_method='infer')
        
    elif params.model_params.model == 'WaveMultimodalInfer_rgb34':
        print('WaveMultimodalInfer_rgb34\n')
        ##changed here for 256 inferall test
        cloud_fe_size = 256 #128
        cloud_fe = MinkLoc(in_channels=1, feature_size=cloud_fe_size, output_dim=cloud_fe_size,
                           planes=[32, 64, 64], layers=[1, 1, 1], num_top_down=1,
                           conv0_kernel_size=5, block='BasicBlock', pooling_method='GeM', net='Wave')
        image_fe_size = 256 #128
        image_fe = TestResnetFPN(out_channels=image_fe_size, lateral_dim=image_fe_size,
                             fh_num_bottom_up=4, fh_num_top_down=0)
        model = MinkLocMultimodal(cloud_fe, cloud_fe_size, image_fe, image_fe_size, output_dim=cloud_fe_size, fuse_method='inferall')
    elif params.model_params.model == 'WaveMultimodalAdd':
        print('Go WaveMultimodalAdd!\n') # original is 128
        cloud_fe_size = 256
        cloud_fe = MinkLoc(in_channels=1, feature_size=cloud_fe_size, output_dim=cloud_fe_size,
                           planes=[32, 64, 64], layers=[1, 1, 1], num_top_down=1,
                           conv0_kernel_size=5, block='BasicBlock', pooling_method='GeM', net='Wave')
        image_fe_size = 256
        image_fe = WaveResnetFPN(out_channels=image_fe_size, lateral_dim=image_fe_size,
                             fh_num_bottom_up=4, fh_num_top_down=0)
        model = MinkLocMultimodal(cloud_fe, cloud_fe_size, image_fe, image_fe_size, output_dim=cloud_fe_size, fuse_method='add')
    elif params.model_params.model == 'WaveMultimodalConcat128':
        print('Go WaveMultimodalConcat128!\n')
        cloud_fe_size = 128
        cloud_fe = MinkLoc(in_channels=1, feature_size=cloud_fe_size, output_dim=cloud_fe_size,
                           planes=[32, 64, 64], layers=[1, 1, 1], num_top_down=1,
                           conv0_kernel_size=5, block='BasicBlock', pooling_method='GeM', net='Wave')
        image_fe_size = 128
        image_fe = WaveResnetFPN(out_channels=image_fe_size, lateral_dim=image_fe_size,
                             fh_num_bottom_up=4, fh_num_top_down=0)
    elif params.model_params.model == 'WaveMultimodalInfer256':
        print('Go WaveMultimodalInfer256!\n')
        cloud_fe_size = 256
        cloud_fe = MinkLoc(in_channels=1, feature_size=cloud_fe_size, output_dim=cloud_fe_size,
                           planes=[32, 64, 64], layers=[1, 1, 1], num_top_down=1,
                           conv0_kernel_size=5, block='BasicBlock', pooling_method='GeM', net='Wave')
        image_fe_size = 256
        image_fe = WaveResnetFPN(out_channels=image_fe_size, lateral_dim=image_fe_size,
                             fh_num_bottom_up=4, fh_num_top_down=0)
        model = MinkLocMultimodal(cloud_fe, cloud_fe_size, image_fe, image_fe_size, output_dim=cloud_fe_size, fuse_method='infer')
    elif params.model_params.model == 'MinkLocMultimodal128':
        print('Go Minkloc-Multimodal128!\n')
        cloud_fe_size = 128
        cloud_fe = MinkLoc(in_channels=1, feature_size=cloud_fe_size, output_dim=cloud_fe_size,
                           planes=[32, 64, 64], layers=[1, 1, 1], num_top_down=1,
                           conv0_kernel_size=5, block='ECABasicBlock', pooling_method='GeM', net='MinkFPN')
        image_fe_size = 128
        ##change: image part
        image_fe = ResnetFPN(out_channels=image_fe_size, lateral_dim=image_fe_size,
                             fh_num_bottom_up=4, fh_num_top_down=0)
        model = MinkLocMultimodal(cloud_fe, cloud_fe_size, image_fe, image_fe_size, output_dim=cloud_fe_size + image_fe_size, final_block='mlp')
    elif params.model_params.model == 'BaseMinkLocMultimodal':
        print('Go Minkloc-Multimodal Base!\n')
        cloud_fe_size = 256
        cloud_fe = MinkLoc(in_channels=1, feature_size=cloud_fe_size, output_dim=cloud_fe_size,
                           planes=[32, 64, 64], layers=[1, 1, 1], num_top_down=1,
                           conv0_kernel_size=5, block='BasicBlock', pooling_method='GeM', net='MinkFPN')
        image_fe_size = 256
        image_fe = ResnetFPN(out_channels=image_fe_size, lateral_dim=image_fe_size,
                             fh_num_bottom_up=4, fh_num_top_down=0)
        model = MinkLocMultimodal(cloud_fe, cloud_fe_size, image_fe, image_fe_size, output_dim=cloud_fe_size, fuse_method='inferall')
    elif params.model_params.model == 'MinkLocMultimodalAdd':
        print('Go Minkloc-MultimodalAdd!\n')
        cloud_fe_size = 128
        cloud_fe = MinkLoc(in_channels=1, feature_size=cloud_fe_size, output_dim=cloud_fe_size,
                           planes=[32, 64, 64], layers=[1, 1, 1], num_top_down=1,
                           conv0_kernel_size=5, block='BasicBlock', pooling_method='GeM', net='MinkFPN')
        image_fe_size = 128
        image_fe = ResnetFPN(out_channels=image_fe_size, lateral_dim=image_fe_size,
                             fh_num_bottom_up=4, fh_num_top_down=0)
        model = MinkLocMultimodal(cloud_fe, cloud_fe_size, image_fe, image_fe_size, output_dim=cloud_fe_size + image_fe_size)
    elif params.model_params.model == 'MinkLocMultimodal':
        print('Go Minkloc-Multimodal!\n')
        cloud_fe_size = 128
        cloud_fe = MinkLoc(in_channels=1, feature_size=cloud_fe_size, output_dim=cloud_fe_size,
                           planes=[32, 64, 64], layers=[1, 1, 1], num_top_down=1,
                           conv0_kernel_size=5, block='ECABasicBlock', pooling_method='GeM', net='MinkFPN')
        image_fe_size = 128
        image_fe = ResnetFPN(out_channels=image_fe_size, lateral_dim=image_fe_size,
                             fh_num_bottom_up=4, fh_num_top_down=0)
        model = MinkLocMultimodal(cloud_fe, cloud_fe_size, image_fe, image_fe_size, output_dim=cloud_fe_size + image_fe_size)
    elif params.model_params.model == 'MinkLoc3D':
        print('Go MinkLoc3D!\n')
        cloud_fe_size = 256
        cloud_fe = MinkLoc(in_channels=1, feature_size=cloud_fe_size, output_dim=cloud_fe_size,
                           planes=[32, 64, 64], layers=[1, 1, 1], num_top_down=1,
                           conv0_kernel_size=5, block='ECABasicBlock', pooling_method='GeM', net='MinkFPN')
        model = MinkLocMultimodal(cloud_fe, cloud_fe_size, None, 0, output_dim=cloud_fe_size,
                                  dropout_p=None)
    elif params.model_params.model == 'MinkLocRGB':
        print('Go MinkLocRGB!\n')
        image_fe_size = 256
        image_fe = ResnetFPN(out_channels=image_fe_size, lateral_dim=image_fe_size,
                             fh_num_bottom_up=4, fh_num_top_down=0)
        model = MinkLocMultimodal(None, 0, image_fe, image_fe_size, output_dim=image_fe_size)
    elif params.model_params.model == 'MinkLocRGB128':
        print('Go MinkLocRGB128!\n')
        image_fe_size = 128
        image_fe = ResnetFPN(out_channels=image_fe_size, lateral_dim=image_fe_size,
                             fh_num_bottom_up=4, fh_num_top_down=0)
        model = MinkLocMultimodal(None, 0, image_fe, image_fe_size, output_dim=image_fe_size)
        
    else:
        raise NotImplementedError('Model not implemented: {}'.format(params.model_params.model))

    return model
