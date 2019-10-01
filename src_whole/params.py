
class Params:

    ################################
    #     dataset
    ################################
    DATA_DIR         = ''

    DATA_TRAIN_TITLE = 'train/train_title.npy'
    DATA_TRAIN_BODY = 'train/train_body.npy'
    DATA_TRAIN_LABEL = 'train/train_label.npy'

    DATA_DEV_TITLE = 'dev/dev_title.npy'
    DATA_DEV_BODY =  'dev/dev_body.npy'
    DATA_DEV_LABEL = 'dev/dev_label.npy'
    
    DATA_TEST_TITLE =  'test/test_title.npy'
    DATA_TEST_BODY =  'test/test_body.npy'
    DATA_TEST_LABEL = 'test/test_label.npy'

    DATA_DEBUG_TITLE = 'debug/debug_title.npy'
    DATA_DEBUG_BODY =  'debug/debug_body.npy'
    DATA_DEBUG_LABEL = 'debug/debug_label.npy'
    
    VOCA_FILE_NAME = 'dic_mincutN.txt'
    GLOVE_FILE_NAME = 'W_embedding.npy'


    ################################
    #     train
    ################################
    till_max_epoch          = False
    num_till_max_epoch      = 8
    
    CAL_ACCURACY_FROM       = 0
    MAX_EARLY_STOP_COUNT    = 5
    EPOCH_PER_VALID_FREQ    = 0.3
    is_embeddign_train      = True     # True is better

    dr_text_in      = 0.3   # 0.3 naacl-18
    dr_text_out     = 1.0
    dr_con_in       = 1.0   # 1.0 naacl-18
    dr_con_out      = 1.0

    ################################
    #     model
    ################################
    
    chunk_tkn_index          = 3    #<EOP>
    pad_index                = 0    #''
    
    reverse_bw               = True
    is_text_encoding_bidir   = False
    is_chunk_encoding_bidir  = True
    
    is_text_residual         = False
    is_chunk_residual        = False
    
    add_attention            = True
    
    
    ################################
    #     etc
    ################################
    IS_DEBUG                 = False     # use short dataset


class Params_NELA18(Params):

    ################################
    #     model
    ################################
    
    chunk_tkn_index          = 131531    #<EOP>
    pad_index                = 0         #''
    
    
    ################################
    #     etc
    ################################
    IS_DEBUG                 = False     # use short dataset