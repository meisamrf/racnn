
import numpy as np

#######################
# Weight rearrange functions
#######################

def mem_align(mem):
    return mem[(16-((mem.ctypes.data%64)//4))%16:]

def realloc_align(mem):
    mem_al = np.zeros((mem.size+16,), mem.dtype)
    idx = (16-((mem_al.ctypes.data%64)//4))%16
    mem_al = mem_al[idx:idx+mem.size]
    mem_al = np.reshape(mem_al, mem.shape)
    np.copyto(mem_al, mem)
    return mem_al
    
def get_bn_weight_bias(wlayer, ix):

    bn_gamma, bn_beta, bn_mean, bn_var = wlayer[ix], wlayer[ix+1], wlayer[ix+2], wlayer[ix+3]
    epsilon = 0.001
    bn_w = bn_gamma / np.sqrt(bn_var + epsilon)
    bn_b = -(bn_w*bn_mean) + bn_beta
    return bn_w, bn_b

def update_weight_with_bn(wlayer, ix):

    bn_w, bn_b = get_bn_weight_bias(wlayer, ix+2)

    if wlayer[ix+0].ndim==4:
        h_w,w_w,d_w,f = wlayer[ix+0].shape
        weight = np.reshape(wlayer[ix+0], (h_w*w_w*d_w,f))*bn_w
    else:
        weight = wlayer[ix+0]*bn_w

    bias = (wlayer[ix+1]*bn_w + bn_b)
    return weight, bias


def nocenter_weight(w_in):

    h, w, d, f = w_in.shape
    w_out = np.zeros((h*w-1, d, f), w_in.dtype)
    idx = 0
    for k in range(9):
        if k==4:
            continue
        y = k//3
        x = k%3
        w_out[idx] = w_in[y,x,:, :]
        idx+=1

    return np.reshape(w_out, ((h*w-1)*d, f))


def get_weight_bycnn(wlayer, ix, extra_dim):
    
    bn_w, bn_b = get_bn_weight_bias(wlayer, ix+6)

    h_w,w_w,d_w,f = wlayer[ix+4].shape
    bias_1x1 = wlayer[ix+5]*bn_w + bn_b

    weight1x1 = np.zeros((h_w*w_w*d_w, f+extra_dim), wlayer[ix+4].dtype)
    weight1x1[:,extra_dim:] = np.reshape(wlayer[ix+4], (h_w*w_w*d_w,f))*bn_w
    weight1x1[:,0] = np.reshape(wlayer[ix+0], (d_w,))

    weight3x3_noc = nocenter_weight(wlayer[ix+2])*bn_w
    bias_3x3 = wlayer[ix+3]*bn_w    
    
    bias_bycnn = wlayer[ix+1][0]
    
    return weight1x1, weight3x3_noc, bias_1x1, bias_3x3, bias_bycnn

def update_weight_dense(wlayer, w_idx_in, w_idx_out, with_bn=False):
    if with_bn:
        weight, bias = update_weight_with_bn(wlayer, w_idx_in)
        w_idx_in += 6
    else:
        weight, bias = wlayer[w_idx_in], wlayer[w_idx_in+1]        
        w_idx_in += 2

    wlayer[w_idx_out] = weight
    wlayer[w_idx_out+1] = bias
    w_idx_out += 2

    return w_idx_out, w_idx_in

def update_weight_conv(wlayer, w_idx_in, w_idx_out, extra_dim, bypass=False, bn=True):
    
    if bypass:
        weight1x1, weight3x3_noc, bias_1x1, bias_3x3, bias_bycnn = get_weight_bycnn(wlayer, w_idx_in, extra_dim)
        wlayer[w_idx_out] = weight1x1
        wlayer[w_idx_out+1] = weight3x3_noc
        wlayer[w_idx_out+2] = bias_1x1
        wlayer[w_idx_out+3] = bias_3x3
        wlayer[w_idx_out+4] = bias_bycnn
        return w_idx_out+5, w_idx_in+10
    else:
        weight, bias = update_weight_with_bn(wlayer, w_idx_in)
        wlayer[w_idx_out] = weight
        wlayer[w_idx_out+1] = bias
        return w_idx_out+2, w_idx_in+6    


def update_weight_with_bn_double(wlayer, w_idx_in, w_idx_out):
    bn_w, bn_b = get_bn_weight_bias(wlayer, w_idx_in+4)
    h_w,w_w,d_w,f = wlayer[w_idx_in].shape
    wlayer[w_idx_out] = np.reshape(wlayer[w_idx_in], (h_w*w_w*d_w,f))*bn_w
    bias_1 = (wlayer[w_idx_in+1]*bn_w + bn_b)

    bn_w, bn_b = get_bn_weight_bias(wlayer, w_idx_in+8)
    h_w,w_w,d_w,f = wlayer[w_idx_in+2].shape
    weight = np.reshape(wlayer[w_idx_in+2], (h_w*w_w*d_w,f))*bn_w
    bias_2 = (wlayer[w_idx_in+3]*bn_w + bn_b)

    bias = bias_1+bias_2
    wlayer[w_idx_out+1] = weight
    wlayer[w_idx_out+2] = bias
    return w_idx_out+3, w_idx_in+12

def update_weight_resnet_conv_block(wlayer, w_idx_in, w_idx_out, extra_dim, bypass=True):
    w_idx_out, w_idx_in = update_weight_conv(wlayer, w_idx_in, w_idx_out, extra_dim)
    w_idx_out, w_idx_in = update_weight_conv(wlayer, w_idx_in, w_idx_out, extra_dim, bypass)
    w_idx_out, w_idx_in = update_weight_with_bn_double(wlayer, w_idx_in, w_idx_out)
    return w_idx_out, w_idx_in

def update_weight_resnet_identity_block(wlayer, w_idx_in, w_idx_out, extra_dim, bypass=True):
    w_idx_out, w_idx_in = update_weight_conv(wlayer, w_idx_in, w_idx_out, extra_dim)
    w_idx_out, w_idx_in = update_weight_conv(wlayer, w_idx_in, w_idx_out, extra_dim, bypass)
    w_idx_out, w_idx_in = update_weight_conv(wlayer, w_idx_in, w_idx_out, extra_dim)
    return w_idx_out, w_idx_in

def update_weight_macro_block(wlayer, w_idx_in, w_idx_out, identity_num, extra_dim, bypass=True):
    w_idx_out, w_idx_in = update_weight_resnet_conv_block(wlayer, w_idx_in, w_idx_out, extra_dim, bypass)
    for k in range(identity_num):
        w_idx_out, w_idx_in = update_weight_resnet_identity_block(wlayer, w_idx_in, w_idx_out, extra_dim, bypass)
    return w_idx_out, w_idx_in


def rearrange_weights(weights, model_type, bypass=False, extra_dim=1):

    '''
    extra_dim considered for mask is vector size for CPU (for vector processing) and 1 for GPU
    '''

    if model_type=='vgg':
        w_idx_in = 0
        w_idx_out = 0
        for k in range(7):
            w_idx_out, w_idx_in = update_weight_conv(weights, w_idx_in, w_idx_out, extra_dim, bypass=bypass)
        for k in range(6):
            w_idx_out, w_idx_in = update_weight_conv(weights, w_idx_in, w_idx_out, extra_dim, bypass=False)
        for k in range(2):
            w_idx_out, w_idx_in = update_weight_dense(weights, w_idx_in, w_idx_out, True)

        w_idx_out, w_idx_in = update_weight_dense(weights, w_idx_in, w_idx_out, False)

    elif model_type=='resnet':
        w_idx_in = 0
        w_idx_out = 0
        w_idx_out, w_idx_in = update_weight_conv(weights, w_idx_in, w_idx_out, extra_dim, bypass=False, bn=True)
        w_idx_out, w_idx_in = update_weight_macro_block(weights, w_idx_in, w_idx_out, 2, extra_dim, bypass=bypass)
        w_idx_out, w_idx_in = update_weight_macro_block(weights, w_idx_in, w_idx_out, 3, extra_dim, bypass=bypass)
        w_idx_out, w_idx_in = update_weight_macro_block(weights, w_idx_in, w_idx_out, 5, extra_dim, bypass=bypass)
        w_idx_out, w_idx_in = update_weight_macro_block(weights, w_idx_in, w_idx_out, 2, extra_dim, bypass=bypass)
        w_idx_out, w_idx_in = update_weight_dense(weights, w_idx_in, w_idx_out)

    for i in range(w_idx_out):
        weights[i] = realloc_align(weights[i])
