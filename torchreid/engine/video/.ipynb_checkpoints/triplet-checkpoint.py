from __future__ import division, print_function, absolute_import
import torch

from torchreid.engine.image import ImageTripletEngine


class VideoTripletEngine(ImageTripletEngine):
    """Triplet-loss engine for video-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        margin (float, optional): margin for triplet loss. Default is 0.3.
        weight_t (float, optional): weight for triplet loss. Default is 1.
        weight_x (float, optional): weight for softmax loss. Default is 1.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_gpu (bool, optional): use gpu. Default is True.
        label_smooth (bool, optional): use label smoothing regularizer. Default is True.
        pooling_method (str, optional): how to pool features for a tracklet.
            Default is "avg" (average). Choices are ["avg", "max"].

    Examples::

        import torch
        import torchreid
        # Each batch contains batch_size*seq_len images
        # Each identity is sampled with num_instances tracklets
        datamanager = torchreid.data.VideoDataManager(
            root='path/to/reid-data',
            sources='mars',
            height=256,
            width=128,
            combineall=False,
            num_instances=4,
            train_sampler='RandomIdentitySampler'
            batch_size=8, # number of tracklets
            seq_len=15 # number of images in each tracklet
        )
        model = torchreid.models.build_model(
            name='resnet50',
            num_classes=datamanager.num_train_pids,
            loss='triplet'
        )
        model = model.cuda()
        optimizer = torchreid.optim.build_optimizer(
            model, optim='adam', lr=0.0003
        )
        scheduler = torchreid.optim.build_lr_scheduler(
            optimizer,
            lr_scheduler='single_step',
            stepsize=20
        )
        engine = torchreid.engine.VideoTripletEngine(
            datamanager, model, optimizer, margin=0.3,
            weight_t=0.7, weight_x=1, scheduler=scheduler,
            pooling_method='avg'
        )
        engine.run(
            max_epoch=60,
            save_dir='log/resnet50-triplet-mars',
            print_freq=10
        )
    """

    def __init__(
        self,
        datamanager,
        model,
        optimizer,
        margin=0.3,
        weight_t=1,
        weight_x=0,#UseLess parameter
        scheduler=None,
        use_gpu=True,
        label_smooth=True,
        pooling_method='avg'
    ):
        super(VideoTripletEngine, self).__init__(
            datamanager,
            model,
            optimizer,
            margin=margin,
            weight_t=weight_t,
            weight_x=weight_x,
            scheduler=scheduler,
            use_gpu=use_gpu,
            label_smooth=label_smooth
        )
        self.pooling_method = pooling_method

    def parse_data_for_train(self, data):
        imgs = data['img']
        pids = data['pid']
        if imgs.dim() == 5:
            # b: batch size
            # s: sqeuence length
            # c: channel depth
            # h: height
            # w: width
            b, s, c, h, w = imgs.size()
            imgs = imgs.view(b * s, c, h, w)
            pids = pids.view(b, 1).expand(b, s)
            pids = pids.contiguous().view(b * s)
        return imgs, pids

    def extract_features(self, input):
        # b: batch size
        # s: sqeuence length
        # c: channel depth
        # h: height
        # w: width
        b, s, c, h, w = input.size()
        input = input.view(b * s, c, h, w)
        outputs, features = [],[]
        if self.model.training == True:
            outputs, features = self.model(input)
        else:
            features = self.model(input)
        #print(features)
        features = features.view(b, s, -1)
        if self.pooling_method == 'avg':
            features = torch.mean(features, 1)
        else:
            features = torch.max(features, 1)[0]
            
        if self.model.training == True:
            return outputs, features
        else:
            return features#outputs?
    
    def forward_backward(self, data):      #new method
        
    
        def prepare(data):
            return data['img'], data['pid']
        
        imgs, pids = prepare(data)
        if self.use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()
            
        #print('pids',pids, 'triplet.py videdo 144')
        outputs, features = self.extract_features(imgs)#BUG on validation
            

        loss = 0
        loss_summary = {}

        if self.weight_t > 0:
            loss_t = self.compute_loss(self.criterion_t, features, pids)
            loss += self.weight_t * loss_t
            loss_summary['loss_t'] = loss_t.item()

        if self.weight_x > 0:
            loss_x = self.compute_loss(self.criterion_x, outputs, pids)
            loss += self.weight_x * loss_x
            loss_summary['loss_x'] = loss_x.item()
            loss_summary['acc'] = metrics.accuracy(outputs, pids)[0].item()
        #print('o')
        assert loss_summary

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss_summary
