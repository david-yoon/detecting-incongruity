
class Params:

    ################################
    #     dataset
    ################################
    DATA_DIR         = '../data/'

    DATA_TRAIN_TITLE = 'train/train_title.npy'
    DATA_TRAIN_BODY = 'train/train_body.npy'
    DATA_TRAIN_LABEL = 'train/train_label.npy'

    DATA_DEV_TITLE = 'dev/dev_title.npy'
    DATA_DEV_BODY =  'dev/dev_body.npy'
    DATA_DEV_LABEL = 'dev/dev_label.npy'
    
    DATA_TEST_TITLE_BODY  = 'test/data_para_test.pkl'
    DATA_TEST_LABEL       = 'test/test_label.npy'

    DATA_DEBUG_TITLE_BODY  = 'debug/data_para_debug.pkl'
    
    
    VOCA_FILE_NAME = 'dic_mincutN.pkl'
    GLOVE_FILE_NAME = 'W_embedding.npy'


    ################################
    #     train
    ################################
    till_max_epoch          = False
    num_till_max_epoch      = 8
    
    CAL_ACCURACY_FROM       = 0
    MAX_EARLY_STOP_COUNT    = 10
    EPOCH_PER_VALID_FREQ    = 0.3
    is_embeddign_train      = True     # True is better
    
    dr_text_in      = 0.3   # 0.3 naacl-18
    dr_text_out     = 1.0
    dr_con_in       = 1.0   # 1.0 naacl-18
    dr_con_out      = 1.0

    ################################
    #     model
    ################################
    reverse_bw               = True
    is_text_encoding_bidir   = False
    is_chunk_encoding_bidir  = True
    
    is_text_residual         = False
    is_chunk_residual        = False
    
    add_attention            = True
    add_LTC                  = False
    LTC_topic_size           = 3
    LTC_memory_dim           = 256
    LTC_dr_prob              = 0.8
    
    
    ################################
    #     etc
    ################################
    IS_DEBUG                 = False     # use short dataset

