def create_model(opt):
    model = None
    #print(opt.model)
    if opt.model == 'cycle_gan':
        assert(opt.dataset_mode == 'unaligned')
        from .cycle_gan_model import CycleGANModel
        model = CycleGANModel()
    elif opt.model == 'pix2pix':
        assert(opt.dataset_mode == 'aligned')
        from .pix2pix_model import Pix2PixModel
        model = Pix2PixModel()
    elif opt.model == 'two_pix2pix':
        if opt.phase == 'train':
            assert(opt.dataset_mode == 'two_aligned')
        elif opt.phase == 'val' or opt.phase == 'test':
            assert(opt.dataset_mode == 'aligned' or opt.dataset_mode == 'single' )
        else:
            print('Warning phase %s' % (opt.phase))
        from .two_pix2pix_model import TwoPix2PixModel
        model = TwoPix2PixModel()
    elif opt.model == 'test':
        assert(opt.dataset_mode == 'single')
        from .test_model import TestModel
        model = TestModel()
        #model = TwoPix2PixModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    #print("model [%s] was created" % (model.name()))
    return model
