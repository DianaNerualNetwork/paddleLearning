import  paddle
import  paddle.nn as  nn
import  paddle.nn.functional as  F

from paddleseg.cvlibs import manager
from paddleseg.utils import utils


class h_sigmoid(nn.Layer):
    def __init__(self):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6()

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Layer):
    def __init__(self):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Layer):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2D((-1, 1))
        self.pool_w = nn.AdaptiveAvgPool2D((1, -1))
        self.sigmoid = nn.Sigmoid()
        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2D(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2D(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2D(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2D(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        n,c,h,w = x.shape
        x_h = nn.AdaptiveAvgPool2D((h, 1))(x)
        x_w = paddle.transpose(nn.AdaptiveAvgPool2D((1, w))(x),[0, 1, 3, 2])
        y = paddle.concat([x_h, x_w], axis=2)

        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = paddle.split(y, [h, w], axis=2)
        x_w = paddle.transpose(x_w,[0, 1, 3, 2])

        a_h = self.sigmoid(self.conv_w(x_h))
        a_w = self.sigmoid(self.conv_w(x_w))

        out = identity * a_w * a_h

        return out

class DAPPM(nn.Layer):
    def __init__(self,in_dim,middle_dim,out_dim):
        super(DAPPM,self).__init__()
        '''
        以1/64图像分辨率的特征图作为输入，采用指数步幅的大池核，生成1/128、1/256、1/512图像分辨率的特征图。
        '''
        kernel_pool=[5,9,17]
        stride_pool=[2,4,8]
        bn_mom = 0.1
        self.scale_1=nn.Sequential(
            nn.AvgPool2D(kernel_size=kernel_pool[0], stride=stride_pool[0], padding=stride_pool[0]),
            nn.BatchNorm2D(in_dim,momentum=bn_mom),
            nn.ReLU(),
            nn.Conv2D(in_dim, middle_dim, kernel_size=1, stride=1, padding=0,bias_attr=False)
        )

        self.scale_2 = nn.Sequential(
            nn.AvgPool2D(kernel_size=kernel_pool[1], stride=stride_pool[1], padding=stride_pool[1]),
            nn.BatchNorm2D(in_dim,momentum=bn_mom),
            nn.ReLU(),
            nn.Conv2D(in_dim, middle_dim, kernel_size=1, stride=1, padding=0,bias_attr=False)
        )

        self.scale_3 = nn.Sequential(
            nn.AvgPool2D(kernel_size=kernel_pool[2], stride=stride_pool[2], padding=stride_pool[2]),
            nn.BatchNorm2D(in_dim,momentum=bn_mom),
            nn.ReLU(),
            nn.Conv2D(in_dim, middle_dim, kernel_size=1, stride=1, padding=0,bias_attr=False)
        )

        self.scale_4 = nn.Sequential(
            nn.AdaptiveAvgPool2D(output_size=(1,1)),
            nn.BatchNorm2D(in_dim, momentum=bn_mom),
            nn.ReLU(),
            nn.Conv2D(in_dim, middle_dim, kernel_size=1, stride=1, padding=0,bias_attr=False)
        )
        #最小扩张感受野
        self.scale_0=nn.Sequential(
            nn.BatchNorm2D(in_dim,momentum=bn_mom),
            nn.ReLU(),
            nn.Conv2D(in_dim,middle_dim,kernel_size=1,stride=1,padding=0,bias_attr=False)
        )
        '''
        残差连接部分
        '''
        self.shortcut=nn.Sequential(
            nn.BatchNorm2D(in_dim,momentum=bn_mom),
            nn.ReLU(),
            nn.Conv2D(in_dim,out_dim,kernel_size=1,stride=1,padding=0,bias_attr=False)
        )
        '''
        conv3x3集合
        '''
        self.process1 = nn.Sequential(
            nn.BatchNorm2D(middle_dim, momentum=bn_mom),
            nn.ReLU(),
            nn.Conv2D(middle_dim, middle_dim, kernel_size=3, padding=1, bias_attr=False),
        )

        self.process2 = nn.Sequential(
            nn.BatchNorm2D(middle_dim, momentum=bn_mom),
            nn.ReLU(),
            nn.Conv2D(middle_dim, middle_dim, kernel_size=3, padding=1, bias_attr=False),
        )

        self.process3 = nn.Sequential(
            nn.BatchNorm2D(middle_dim, momentum=bn_mom),
            nn.ReLU(),
            nn.Conv2D(middle_dim, middle_dim, kernel_size=3, padding=1, bias_attr=False),
        )

        self.process4 = nn.Sequential(
            nn.BatchNorm2D(middle_dim, momentum=bn_mom),
            nn.ReLU(),
            nn.Conv2D(middle_dim, middle_dim, kernel_size=3, padding=1, bias_attr=False),
        )

        self.concat_conv1x1=nn.Sequential(
            nn.BatchNorm2D(middle_dim * 5, momentum=bn_mom),
            nn.ReLU(),
            nn.Conv2D(middle_dim * 5, out_dim, kernel_size=1, bias_attr=False),
        )

    def forward(self,inps):

        layer_list=[]
        H=8
        W=15

        '''
            {   C1x1(x)                                     i=1
        Y_i=|   C3x3( U(  C1x1(AvgPool) )  + Y_i-1  )       1<i<n
            {   C3x3( U(  C1x1(GlobalPool) )  + Y_i-1  )      i=n
        '''
        layer_list.append(self.scale_0(inps))
        layer_list.append(self.process1(F.interpolate( self.scale_1(inps),
                                                    #   scale_factor=[2, 2],
                                                        size=[H,W],
                                                       mode='bilinear')
                                +layer_list[0]))
        layer_list.append(self.process2(F.interpolate(self.scale_2(inps),
                                                    #  scale_factor=[4, 4],
                                                       size=[H, W],
                                                      mode='bilinear')
                          + layer_list[1]))
        layer_list.append(self.process3(F.interpolate(self.scale_3(inps),
                                                    #  scale_factor=[8, 8],
                                                       size=[H, W],
                                                      mode='bilinear')
                          + layer_list[2]))
        layer_list.append(self.process4(F.interpolate(self.scale_4(inps),
                                                    #  scale_factor=[16, 8],
                                                       size=[H, W],
                                                      mode='bilinear')
                         + layer_list[3]))
        #concat
        oups=self.concat_conv1x1(paddle.concat(layer_list,axis=1))+self.shortcut(inps)

        return  oups

class BasicBlock(nn.Layer):

    def __init__(self, in_dim, out_dim, stride=1, downsample=None, no_relu=False):
        super(BasicBlock,self).__init__()
        '''
        组成低分辨率和高分辨率分支的卷积集合块，由两个3x3卷积组成
        '''
        bn_mom=0.1
        self.expansion = 1.0
        self.layer1 = nn.Sequential(
            nn.Conv2D(in_dim,out_dim, kernel_size=3, stride=stride, padding=1, bias_attr=False),
            nn.BatchNorm2D(out_dim, momentum=bn_mom),
            nn.ReLU(),
            nn.Conv2D(out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias_attr=False),
            nn.BatchNorm2D(out_dim, momentum=bn_mom)
        )
        self.attention=CoordAtt(inp=out_dim,oup=out_dim)
        self.layer2=nn.Sequential(
            nn.Conv2D(out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias_attr=False),
            nn.BatchNorm2D(out_dim, momentum=bn_mom)
        )

        self.no_relu = no_relu
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, inps):
        residual=inps
        oups=self.layer1(inps)
        oups=self.layer2(oups)
        #oups=self.attention(oups)
        if self.downsample is not None:
            residual=self.downsample(inps)
        oups=residual+oups
        if self.no_relu:
            oups=oups
        else:
            oups = self.relu(oups)
        return  oups

class BottleNeckBlock(nn.Layer):

    def __init__(self, in_dim, mid_dim, stride=1, downsample=None, no_relu=True):
        super(BottleNeckBlock, self).__init__()
        self.expansion = 2
        self.layer1=nn.Sequential(
            nn.Conv2D(in_dim, mid_dim, kernel_size=1, bias_attr=False),
            nn.BatchNorm2D(mid_dim,momentum=0.1),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv2D(mid_dim, mid_dim, kernel_size=3, stride=stride, padding=1, bias_attr=False),
            nn.BatchNorm2D(mid_dim, momentum=0.1),
            nn.ReLU()

        )
        self.attention = CoordAtt(inp=mid_dim * self.expansion, oup=mid_dim * self.expansion)
        self.layer3 = nn.Sequential(
            nn.Conv2D(mid_dim, mid_dim * self.expansion, kernel_size=1, bias_attr=False),
            nn.BatchNorm2D(mid_dim * self.expansion, momentum=0.1),
            nn.ReLU(),

        )

        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x
        out=self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        #out=self.attention(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)

class Stem(nn.Layer):
    def __init__(self,in_dim,out_dim):
        super(Stem,self).__init__()
        '''
        进行两次下采样
        '''
        bn_mom=0.1
        self.layer=nn.Sequential(
            nn.Conv2D(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias_attr=False),
            nn.BatchNorm2D(out_dim, momentum=bn_mom),
            nn.ReLU(),
            nn.Conv2D(out_dim, out_dim, kernel_size=3, stride=2, padding=1, bias_attr=False),
            nn.BatchNorm2D(out_dim, momentum=bn_mom)
        )
    def forward(self,inps):
        oups=self.layer(inps)
        return oups

class SegHead(nn.Layer):
    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(SegHead, self).__init__()
        self.layer=nn.Sequential(
            nn.BatchNorm2D(inplanes,momentum=0.1),
            nn.Conv2D(inplanes,interplanes,kernel_size=3,padding=1,bias_attr=False),
            nn.ReLU(),
            nn.BatchNorm2D(interplanes,momentum=0.1),
            nn.ReLU(),
            nn.Conv2D(interplanes,outplanes,kernel_size=1,padding=0,bias_attr=True)
        )

        self.scale_factor = scale_factor
    def forward(self,inps):
        oups=self.layer(inps)

        if self.scale_factor is not None:
            height = inps.shape[-2] * self.scale_factor
            width = inps.shape[-1] * self.scale_factor
            oups = F.interpolate(oups,
                                size=[height, width],
                                mode='bilinear')

        return  oups

def _make_layer(block,in_dim,out_dim,block_num,stride=1,expansion=1):
    layer=[]
    downsample=None
    if stride!=1 or  in_dim != out_dim * expansion:
        downsample = nn.Sequential(
            nn.Conv2D(in_dim, out_dim * expansion,kernel_size=1, stride=stride, bias_attr=False),
            nn.BatchNorm2D(out_dim * expansion, momentum=0.1),
        )
    layer.append(block(in_dim, out_dim, stride, downsample))
    inplanes = out_dim * expansion
    for i in range(1, block_num):
        if i == (block_num - 1):
            layer.append(block(inplanes, out_dim, stride=1, no_relu=True))
        else:
            layer.append(block(inplanes, out_dim, stride=1, no_relu=False))
    return nn.Sequential(*layer)

@manager.MODELS.add_component
class DDRNet_23_slim(nn.Layer):
    def __init__(self,num_classes=4):
        super(DDRNet_23_slim,self).__init__()
        spp_plane=128
        head_plane=64
        planes=32
        highres_planes=planes*2
        basicBlock_expansion=1
        bottleNeck_expansion=2
        self.relu=nn.ReLU()
        self.stem=Stem(3,32)
        self.layer1 = _make_layer(BasicBlock, planes, planes, block_num=2,stride=1 ,expansion=basicBlock_expansion)
        self.layer2 = _make_layer(BasicBlock, planes, planes * 2, block_num=2, stride=2,expansion=basicBlock_expansion)
        #1/8下采样
        self.layer3 = _make_layer(BasicBlock, planes * 2, planes * 4, block_num=2, stride=2,expansion=basicBlock_expansion)
        #1/16下采样 高分辨率
        self.layer3_ =_make_layer(BasicBlock, planes * 2, highres_planes, 2,stride=1,expansion=basicBlock_expansion)

        self.layer4 = _make_layer(BasicBlock, planes * 4, planes * 8, block_num=2, stride=2,expansion=basicBlock_expansion)
        #high-resolution branch layer attch to layer3_
        self.layer4_ = _make_layer(BasicBlock, highres_planes, highres_planes, 2,stride=1,expansion=basicBlock_expansion)

        '''
        上采样过程中的操作
        '''
        self.compression3 = nn.Sequential(
            nn.Conv2D(planes * 4, highres_planes, kernel_size=1, bias_attr=False),
            nn.BatchNorm2D(highres_planes, momentum=0.1),
        )

        self.compression4 = nn.Sequential(
            nn.Conv2D(planes * 8, highres_planes, kernel_size=1, bias_attr=False),
            nn.BatchNorm2D(highres_planes, momentum=0.1),
        )

        '''
        low-resolution brach downsample
        '''
        self.down3 = nn.Sequential(
            nn.Conv2D(highres_planes, planes * 4, kernel_size=3, stride=2, padding=1, bias_attr=False),
            nn.BatchNorm2D(planes * 4, momentum=0.1),
        )

        self.down4 = nn.Sequential(
            nn.Conv2D(highres_planes, planes * 4, kernel_size=3, stride=2, padding=1, bias_attr=False),
            nn.BatchNorm2D(planes * 4, momentum=0.1),
            nn.ReLU(),
            nn.Conv2D(planes * 4, planes * 8, kernel_size=3, stride=2, padding=1, bias_attr=False),
            nn.BatchNorm2D(planes * 8, momentum=0.1),
        )

        self.layer5_ = _make_layer(BottleNeckBlock, highres_planes, highres_planes, 1,stride=1,expansion=bottleNeck_expansion)

        self.layer5 = _make_layer(BottleNeckBlock, planes * 8, planes * 8, 1, stride=2,expansion=bottleNeck_expansion)

        self.spp = DAPPM(planes * 16, spp_plane, planes * 4)

        self.final_layer = SegHead(planes * 4, head_plane, num_classes)


    def forward(self, inps):
        layers = [0, 0, 0, 0]
        # layers = []
        # width_output = inps.shape[-1] // 8
        # height_output = inps.shape[-2] // 8
        x = self.stem(inps)

        x = self.layer1(x)
        layers[0] = x # -- 0
        # layers.append(x)
        x = self.layer2(self.relu(x))
        layers[1] = x #1/8 -- 1
        # layers.append(x)
        #Bilateral fusion
        x = self.layer3(self.relu(x)) # get 1/16
        layers[2] = x # -- 2
        # layers.append(x)
        x_low_resolution_branch = self.layer3_(self.relu(layers[1])) #get 1/8
        x = x + self.down3(self.relu(x_low_resolution_branch))  # 低分辨度下采样成1/16+高分辨率1/16
        x_low_resolution_branch = x_low_resolution_branch + F.interpolate(
            self.compression3(self.relu(layers[2])),
            scale_factor=[2, 2], # from 1/16 to 1/8
            # size=[height_output, width_output],
            mode='bilinear') #低分辨率1/16上采样成高分辨率的1/8
        x = self.layer4(self.relu(x)) # 1/32
        layers[3] = x # -- 3
        # layers.append(x)
        x_ = self.layer4_(self.relu(x_low_resolution_branch)) # 1/8

        x = x + self.down4(self.relu(x_)) # 1/32
        x_ = x_ + F.interpolate(
            self.compression4(self.relu(layers[3])),
            scale_factor=[4, 4], # from 1/32 to 1/8
            # size=[height_output, width_output],
            mode='bilinear') # 1/8

        x_ = self.layer5_(self.relu(x_)) # 1/8
        x = F.interpolate(
            self.spp(self.layer5(self.relu(x))),
            #scale_factor=[8, 8], # from 1/64 to 1/8
            size=[60, 120],
            mode='bilinear')

        x_ = self.final_layer(x + x_) # 1/8
        oups=F.interpolate(x_,
                           scale_factor=[8, 8],
                        #    size=[inps.shape[-2],inps.shape[-1]],
                           mode='bilinear')
        return [oups]