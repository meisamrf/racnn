import numpy as np
import sys
import os
import time
import racnnlib
import pickle
import racnn_utils

def softmax(input):
    max_val = np.max(input)
    exp_val= np.exp(input-max_val)
    softmax_val = exp_val/np.sum(exp_val)
    return softmax_val

###############################################
#  CNN & DENSE LAYERS IMPLEMENTATION
###############################################

class RACNNLayer():

    def conv1x1(input_shape, buffer_in, buffer_out, weight, bias):
        h,w,d = input_shape
        in_mat = np.reshape(buffer_in[0:h*w*d], (h*w,d))
        out_mat = np.reshape(buffer_out[0: (h*w*weight.shape[1])], (h*w,weight.shape[1]))
        np.matmul(in_mat, weight, out = out_mat)
        if (bias is not None):
            racnnlib.bias_relu(out_mat, bias)
            #np.fmax(out_mat + bias,0, out = out_mat)  
        outsize = h,w,weight.shape[1]
        return outsize

    def conv3x3(in_shape, buf_in, buf_out, mat_buf, weight, bias, pool=False):

        h, w, d = in_shape
        inp = np.reshape(buf_in[0:h*w*d], (h, w, d))
        #Kernel size
        ks = 3
        inp_mat = np.reshape(mat_buf[0:h*w*d*ks*ks], (h*w, d*ks*ks))
        racnnlib.im2col(inp, inp_mat, ks) 

        if (not pool):
            out = np.reshape(buf_out[0:inp_mat.shape[0]*weight.shape[1]], (inp_mat.shape[0], weight.shape[1]))                        
            np.matmul(inp_mat,weight,out=out)
            h, w, d = (inp.shape[0], inp.shape[1], out.shape[1])
            if (bias is not None):
                out = np.reshape(out, (h, w, d))
                racnnlib.bias_relu(out, bias)
        else:
            mul_out = np.reshape(mat_buf[0:inp_mat.shape[0]*weight.shape[1]], (inp_mat.shape[0], weight.shape[1]))
            np.matmul(inp_mat,weight,out=mul_out)
            mul_out = np.reshape(mul_out, (h, w, weight.shape[1]))                        
            h, w, d = (h//2, w//2, weight.shape[1])
            pool_out = np.reshape(buf_out[0:h*w*d], (h, w, d))
            racnnlib.bias_relu(mul_out, bias, pool_out, 2)

        return (h, w, d)


    def conv3x3_racnn(buffer_in, buffer_out1x1, buffer_out3x3, inp_shape, weight1x1, 
                   weight3x3_noc, bias_1x1, bias_3x3, bias_mask, pool=False):  
        
        h,w,d = inp_shape
        in_mat = np.reshape(buffer_in[0:h*w*d], (h,w,d))
        out1x1 = np.reshape(buffer_out1x1[0:(h*w*weight1x1.shape[1])], (h*w,weight1x1.shape[1]))
        out_mat3x3 = np.reshape(buffer_out3x3[0: h*w*8*d], (h*w, 8*d))

        np.matmul(np.reshape(in_mat, (h*w,d)), weight1x1, out=out1x1)
        mask_z = out1x1[:,0]
        
        count = racnnlib.im2col(in_mat, out_mat3x3, 3, mask_z, bias_mask)
        
        outmul_noc = np.reshape(buffer_in[0: count*weight3x3_noc.shape[1]], (count, weight3x3_noc.shape[1]))
        np.matmul(out_mat3x3[0:count], weight3x3_noc, out = outmul_noc)

        dim = out1x1.shape[1]-8
        img_out = np.reshape(buffer_out3x3[0: out1x1.shape[0]*(out1x1.shape[1]-8)], (out1x1.shape[0], dim))

        if pool:                
            outsize = h//2, w//2, dim
            img_out = np.reshape(buffer_out3x3[0: (h//2)*(w//2)*dim], (h//2, w//2, dim))
            racnnlib.col2im_mask(outmul_noc, out1x1, bias_3x3, bias_1x1, img_out, 2)                
            return outsize
        else:
            racnnlib.col2im_mask(outmul_noc, out1x1, bias_3x3, bias_1x1, img_out)
        
        
        outsize = h,w,bias_3x3.shape[0]
        
        return outsize

    def dense(inp_shape, buffer_in, buffer_out, wlayer, ix, activation='relu'):
        if len(inp_shape)==3:
            h,w,d = inp_shape
        else:
            h = inp_shape[0]
            w = 1
            d = 1
        weight = wlayer[ix]
        bias  = wlayer[ix+1]
        out_mat = np.reshape(buffer_out[0: weight.shape[1]], (weight.shape[1],))
        np.matmul(buffer_in[0:h*w*d], weight, out=out_mat)
        if activation=='relu':
            np.fmax(out_mat + bias,0, out = out_mat)
        elif activation=='softmax':
            np.copyto(out_mat, softmax(out_mat+bias))
        
        return (weight.shape[1],), ix+2

    def conv7x7_rgb(inp, input_buf, output_buf, buffer_tmp, wlayer, ix):
        """
        conv2d 7x7 with stride = 2 and RGB input
        """
        weight = wlayer[ix]
        bias  = wlayer[ix+1]
        h,w,d = (inp.shape[0]//2), (inp.shape[1]//2), inp.shape[2]*49
        mat_out = np.reshape(input_buf[0:h*w*d], (h*w,d))
        racnnlib.im2col(inp, mat_out, 7)
        mul_out = np.reshape(buffer_tmp[0:h*w*bias.shape[0]], (h*w,bias.shape[0]))
        np.matmul(mat_out, weight, out = mul_out)
        pool_out = np.reshape(output_buf[0:(h//2)*(w//2)*bias.shape[0]], ((h//2),(w//2),bias.shape[0]))
        mul_out = np.reshape(mul_out, (h, w, bias.shape[0]))
        racnnlib.bias_relu(mul_out, bias, pool_out, 3)
        outsize = h//2, w//2, bias.shape[0]
        return outsize, ix+2

    def avgpool_dense_softmax(inp_shape, buffer_in, buffer_out, buffer1, wlayer, ix):
        weight = wlayer[ix]
        bias  = wlayer[ix+1]
        h,w,d = inp_shape
        in_mat = np.reshape(buffer_in[0:h*w*d], (h,w,d))
        out_pool = np.reshape(buffer1[0: ((h//2)*(w//2)*d)], ((h//2),(w//2),d))  
        racnnlib.avg_pool2(in_mat, out_pool)
        h,w = h//2,w//2
        out_mat = np.reshape(buffer_out[0: weight.shape[1]], (weight.shape[1],))
        np.matmul(buffer1[0:h*w*d], weight, out=out_mat)
        np.copyto(out_mat, softmax(out_mat+bias))
        return weight.shape[1],ix+2

    def resample2(inp_shape, buffer_in, buffer_out):
        h,w,d = inp_shape
        in_mat = np.reshape(buffer_in[0:h*w*d], (h,w,d))
        h, w = h//2, w//2
        out_mat = np.reshape(buffer_out[0:h*w*d], (h,w,d))
        np.copyto(out_mat, in_mat[0::2,0::2,:])        
        return (h,w,d)

class vgg16_racnn():
    """
    """

    def __init__(self, weight_file, bypass=True):
        self.weights = pickle.load(open(weight_file,"rb"))
        self.model_shape = (224,224,3)
        self.bypass = bypass
        h,w,c = self.model_shape
        if bypass:
            self.buffer1 = np.zeros((h*w*64*8+16,), np.float32)
            self.buffer1 = racnn_utils.mem_align(self.buffer1)
            self.buffer2 = np.zeros((h*w*64*4+16,), np.float32)
            self.buffer2 = racnn_utils.mem_align(self.buffer2)
            self.buffer3 = np.zeros((h*w*8*9+16, ), np.float32)
            self.buffer3 = racnn_utils.mem_align(self.buffer3)
        else:
            self.buffer1 = np.zeros((h*w*64*9+16,), np.float32)
            self.buffer1 = racnn_utils.mem_align(self.buffer1)
            self.buffer2 = np.zeros((h*w*64*4+16,), np.float32)
            self.buffer2 = racnn_utils.mem_align(self.buffer2)
            self.buffer3 = np.zeros((h*w*64*9+16, ), np.float32)
            self.buffer3 = racnn_utils.mem_align(self.buffer3)

        racnn_utils.rearrange_weights(self.weights, 'vgg', bypass=bypass, extra_dim=8)

    def vgg_conv_block_bypass(input_size, buffer_in, buffer_out, buffer_tmp, wlayer, ix, 
                              block_count):

        buf1 = buffer_in
        buf2 = buffer_out
        outsize = input_size
        for k in range(block_count):
            outsize = RACNNLayer.conv3x3_racnn(buf1, buffer_tmp, buf2, outsize, wlayer[ix+0], 
                       wlayer[ix+1], wlayer[ix+2], wlayer[ix+3], wlayer[ix+4], k==block_count-1)
            bufx = buf1
            buf1 = buf2
            buf2 = bufx
            ix+=5

        return outsize, ix

    def vgg_conv_block(input_size, buffer_in, buffer_out, buffer_tmp, wlayer, ix, block_count):

        buf1 = buffer_in
        buf2 = buffer_out
        outsize = input_size

        for k in range(block_count):
            outsize = RACNNLayer.conv3x3(outsize, buf1, buf2, buffer_tmp, wlayer[ix+0], 
                       wlayer[ix+1], k==block_count-1)
            bufx = buf1
            buf1 = buf2
            buf2 = bufx
            ix+=2

        return outsize, ix   


    def predict(self, input_img):

        iosize = input_img.shape
        wcnt = 0
        np.copyto(self.buffer1[0:iosize[0]*iosize[1]*iosize[2]], input_img.flatten())

        if not self.bypass:
            # Stage 1
            iosize, wcnt  = vgg16_racnn.vgg_conv_block(iosize, self.buffer1, self.buffer2, self.buffer3, 
                                                  self.weights, wcnt, block_count=2)
            # Stage 2
            iosize, wcnt  = vgg16_racnn.vgg_conv_block(iosize, self.buffer1, self.buffer2, self.buffer3, 
                                                  self.weights, wcnt, block_count=2)
            # Stage 3
            iosize, wcnt  = vgg16_racnn.vgg_conv_block(iosize, self.buffer1, self.buffer2, self.buffer3, 
                                                  self.weights, wcnt, block_count=3)
            buf_i = self.buffer2
            buf_o = self.buffer1

        else:
            # Stage 1
            iosize, wcnt  = vgg16_racnn.vgg_conv_block_bypass(iosize, self.buffer1, self.buffer2, self.buffer3, 
                                                    self.weights, wcnt, block_count=2)
            # Stage 2
            iosize, wcnt  = vgg16_racnn.vgg_conv_block_bypass(iosize, self.buffer1, self.buffer2, self.buffer3, 
                                                    self.weights, wcnt, block_count=2)
            # Stage 3
            iosize, wcnt  = vgg16_racnn.vgg_conv_block_bypass(iosize, self.buffer1, self.buffer2, self.buffer3, 
                                                    self.weights, wcnt, block_count=3)
            buf_i = self.buffer2
            buf_o = self.buffer1


        # Stage 4
        iosize, wcnt  = vgg16_racnn.vgg_conv_block(iosize, buf_i, buf_o, self.buffer3, 
                                        self.weights, wcnt, block_count=3)
        # Stage 5
        iosize, wcnt  = vgg16_racnn.vgg_conv_block(iosize, buf_o, buf_i, self.buffer3, 
                                        self.weights, wcnt, block_count=3)
        # Stage 6
        iosize, wcnt  = RACNNLayer.dense(iosize, buf_i, buf_o, self.weights, wcnt)
        # Stage 7
        iosize, wcnt  = RACNNLayer.dense(iosize, buf_o, buf_i, self.weights, wcnt)
        # Stage 8
        iosize, wcnt  = RACNNLayer.dense(iosize, buf_i, buf_o, self.weights, wcnt, 'softmax')

        return np.reshape(buf_o[0: iosize[0]], iosize)


class resnet50_racnn():
    """
    """
    def __init__(self, weight_file, bypass=True):
        self.weights = pickle.load(open(weight_file,"rb"))
        self.model_shape = (256,256,3)
        self.bypass = bypass

        h,w,c = self.model_shape
        if bypass:
            self.buffer1 = np.zeros(((h//2)*(w//2)*c*49+16,), np.float32)
            self.buffer1 = racnn_utils.mem_align(self.buffer1)
            self.buffer2 = np.zeros(((h//2)*(w//2)*64+16,), np.float32)
            self.buffer2 = racnn_utils.mem_align(self.buffer2)
            self.buffer3 = np.zeros(((h//4)*(w//4)*(64+8)+16, ), np.float32)
            self.buffer3 = racnn_utils.mem_align(self.buffer3)
            self.buffer4 = np.zeros(((h//2)*(w//2)*64+16, ), np.float32)
            self.buffer4 = racnn_utils.mem_align(self.buffer4)
        else:
            self.buffer1 = np.zeros(((h//2)*(w//2)*c*49+16,), np.float32)
            self.buffer1 = racnn_utils.mem_align(self.buffer1)
            self.buffer2 = np.zeros(((h//2)*(w//2)*64+16,), np.float32)
            self.buffer2 = racnn_utils.mem_align(self.buffer2)
            self.buffer3 = np.zeros(((h//4)*(w//4)*64+16, ), np.float32)
            self.buffer3 = racnn_utils.mem_align(self.buffer3)
            self.buffer4 = np.zeros(((h//2)*(w//2)*64+16, ), np.float32)
            self.buffer4 = racnn_utils.mem_align(self.buffer4)

        racnn_utils.rearrange_weights(self.weights, 'resnet', bypass=bypass, extra_dim=8)

    def resnet_conv_block_racnn(input_size, shortcut_b, buffer1, buffer2, buffer3, wlayer, ix, 
                                 bypass=True):

        # conv2d A 
        outsize_a = RACNNLayer.conv1x1(input_size, shortcut_b, buffer2, wlayer[ix+0], wlayer[ix+1])

        # conv2d B 'Proposed Idea'
        if bypass:
            outsize_b = RACNNLayer.conv3x3_racnn(buffer2, buffer3, buffer1, outsize_a, wlayer[ix+2], 
                           wlayer[ix+3], wlayer[ix+4], wlayer[ix+5], wlayer[ix+6], )
            ix+=7
            # conv2d C 
            outsize_c = RACNNLayer.conv1x1(outsize_b, buffer1, buffer2, wlayer[ix], None)
        else:
            outsize_b = RACNNLayer.conv3x3(outsize_a, buffer2, buffer3, buffer1, wlayer[ix+2], wlayer[ix+3])
            ix+=4
            # conv2d C 
            outsize_c = RACNNLayer.conv1x1(outsize_b, buffer3, buffer2, wlayer[ix], None)

        # conv2d Shortcut
        outsize_shortcut = RACNNLayer.conv1x1(input_size, shortcut_b, buffer1, wlayer[ix+1], None)

        h,w,d = outsize_c
        mat_out = np.reshape(buffer2[0: (h*w*d)], (h*w,d))
        h,w,d = outsize_shortcut
        short_out = np.reshape(buffer1[0: (h*w*d)], (h*w,d))

        racnnlib.add_bias_relu(mat_out, short_out, wlayer[ix+2])

        #np.add(mat_out, short_out, out=mat_out)
        #np.fmax(mat_out + wlayer[ix+2],0, out = mat_out)

        return outsize_shortcut, ix+3


    def resnet_identity_block_racnn(input_size, shortcut_b, buffer1, buffer2, buffer3, wlayer, 
                                     ix, bypass=True):

        # conv2d A 
        outsize_a = RACNNLayer.conv1x1(input_size, shortcut_b, buffer2, wlayer[ix+0], wlayer[ix+1])

        # conv2d B 'Proposed Idea'
        if bypass:
            outsize_b = RACNNLayer.conv3x3_racnn(buffer2, buffer3, buffer1, outsize_a, wlayer[ix+2], 
                       wlayer[ix+3], wlayer[ix+4], wlayer[ix+5], wlayer[ix+6])
            ix+=7
            # conv2d C 
            outsize_c = RACNNLayer.conv1x1(outsize_b, buffer1, buffer2, wlayer[ix], None)
        else:
            outsize_b = RACNNLayer.conv3x3(outsize_a, buffer2, buffer3, buffer1,  wlayer[ix+2], wlayer[ix+3])
            ix+=4
            # conv2d C 
            outsize_c = RACNNLayer.conv1x1(outsize_b, buffer3, buffer2, wlayer[ix], None)


        # conv2d Shortcut
        h,w,d = outsize_c
        mat_out = np.reshape(buffer2[0: (h*w*d)], (h*w,d))    
        short_out = np.reshape(shortcut_b[0: (h*w*d)], (h*w,d))

        racnnlib.add_bias_relu(mat_out, short_out, wlayer[ix+1])

        return outsize_c, ix+2


    def resnet_macro_block(input_size, shortcut, buffer1, buffer2, buffer3, wlayer, wcnt, block_count, 
                           bypass=True):

        iosize = input_size
        iosize, wcnt  = resnet50_racnn.resnet_conv_block_racnn(iosize, shortcut, buffer1, buffer2, 
                                                    buffer3, wlayer, wcnt, bypass=bypass)
        buf_i = buffer2
        buf_o = shortcut
        for k in range(block_count):
            iosize, wcnt = resnet50_racnn.resnet_identity_block_racnn(iosize, buf_i, buffer1, buf_o, buffer3,
                                                                      wlayer, wcnt, bypass)        
            bufx = buf_i
            buf_i = buf_o
            buf_o = bufx  
        
        return iosize, wcnt

    def predict(self, input_img):
        """
        """        
        wcnt = 0
        # Stage 1
        iosize, wcnt = RACNNLayer.conv7x7_rgb(input_img, self.buffer1, self.buffer4, self.buffer2, 
                                              self.weights, wcnt)
        # Stage 2
        iosize, wcnt  = resnet50_racnn.resnet_macro_block(iosize, self.buffer4, self.buffer1, self.buffer2, 
                                                 self.buffer3, self.weights, wcnt, 2, 
                                                 bypass=self.bypass)

        iosize = RACNNLayer.resample2(iosize, self.buffer2, self.buffer4)

        iosize, wcnt  = resnet50_racnn.resnet_macro_block(iosize, self.buffer4, self.buffer1, self.buffer2, 
                                                 self.buffer3, self.weights, wcnt, 3, 
                                                 bypass=self.bypass)
    
        iosize = RACNNLayer.resample2(iosize, self.buffer4, self.buffer2)

        iosize, wcnt  = resnet50_racnn.resnet_macro_block(iosize, self.buffer2, self.buffer1, self.buffer4, 
                                                 self.buffer3, self.weights, wcnt, 5, 
                                                 bypass=self.bypass)

        iosize = RACNNLayer.resample2(iosize, self.buffer2, self.buffer4)

        iosize, wcnt  = resnet50_racnn.resnet_macro_block(iosize, self.buffer4, self.buffer1, self.buffer2, 
                                                 self.buffer3, self.weights, wcnt, 2, 
                                                 bypass=self.bypass)

        iosize, wcnt = RACNNLayer.avgpool_dense_softmax(iosize, self.buffer2, self.buffer1, self.buffer4, 
                                                        self.weights, wcnt)

        return self.buffer1[0:iosize]




