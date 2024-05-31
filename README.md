# DATA620004_HW2_FRCNN-YOLOV3_mmdetection

复旦大学  神经网络和深度学习  
作业2 任务2：在 VOC 数据集上训练并测试目标检测模型 Faster R-CNN 和 YOLO V3

## 项目描述
本任务旨在学习使用现成的目标检测框架（如 mmdetection 或 detectron2）在 VOC 数据集上训练并测试目标检测模型 Faster R-CNN 和 YOLO V3。主要步骤包括：

1. **使用目标检测框架**：在 VOC 数据集上训练并测试目标检测模型 Faster R-CNN 和 YOLO V3。
2. **可视化比较**：挑选测试集中的图像，对比 Faster R-CNN 第一阶段产生的 proposal box 和最终的预测结果。
3. **外部图像测试**：搜集三张不在 VOC 数据集内的图像，展示并比较两个模型的检测结果。

## 安装指南

### 安装 mmdetection
请参考 mmdetection 的 GitHub 页面或官方文档进行安装：
- [MMDetection GitHub](https://github.com/open-mmlab/mmdetection)
- [MMDetection 官方文档](https://mmdetection.readthedocs.io/zh-cn/latest/get_started.html)

### 安装其他依赖
```bash
pip install tensorboard tensorboardX
pip install torch==1.2.0
```

### 数据准备
下载并解压 PASCAL VOC2007 和 2012 数据集到以下目录：
```
./vocdata
```

## 训练模型

### 训练 Faster R-CNN
使用 `train_frcnn.sh` 脚本进行训练。训练参数和数据集路径可在以下配置文件中修改：
```
./configs/faster_rcnn/faster-rcnn_r50_fpn_2x_coco.py
```
训练结果及参数配置详见：
```
./output_0528_frcnn2x/faster-rcnn_r50_fpn_2x_coco.py
```

### 训练 YOLOV3
使用 `train_yolov3.sh` 脚本进行训练。训练参数和数据集路径可在以下配置文件中修改：
```
./configs/yolo/yolov3_d53_8xb8-ms-416-273e_coco.py
```
训练结果及参数配置详见：
```
./output_test_wyq_yolov3/yolov3_d53_8xb8-ms-416-273e_coco.py
```

## 单张图片检测

使用 `infer_single_pic.sh` 脚本进行单张图片检测。请确保正确设置图像路径、配置文件、权重文件和输出目录。示例命令：

```bash
CUDA_VISIBLE_DEVICES=6 python ./demo/image_demo_try.py $image_path $config_file --weights $weights_file --out-dir $output_dir
```
$image_path:检测图片路径

$config_file：训练模型output中的config文件

$weights_file：训练好的权重文件

$output_dir：输出目录

如果需要获取Faster R-CNN第一阶段的proposal boxes，请修改./mmdet/models/detectors/two_stage.py中，239-265行取消注释并修改路径

对应代码为：

```bash
import torch
import mmcv
from mmengine.visualization import Visualizer

# 加载你的图像文件，确保路径正确
image_name = "000001.jpg"
image_path = f'yourpath/{image_name}'  # 修改为你的图像路径
image = mmcv.imread(image_path, channel_order='rgb')
scale = min(1000/ image.shape[1], 600 / image.shape[0])
print(image.shape)
scale_x = scale
scale_y = scale
bboxes_ori = rpn_results_list[0].bboxes[:20]
bboxes = bboxes_ori.clone()
bboxes[:, [0, 2]] /= scale_x  # 调整x坐标
bboxes[:, [1, 3]] /= scale_y  # 调整y坐标
# 创建Visualizer实例
visualizer = Visualizer(image=image)

# 绘制所有检测框
visualizer.draw_bboxes(bboxes, edge_colors="b")
drawn_image = visualizer.get_image()
# 展示图像
output_path = f'yourpath/mmdetection/output_proposal/{image_name}'  # 设置输出文件的路径
mmcv.imwrite(drawn_image, output_path)  # 保存图像
```


## 测试模型
使用 `test_yolov3_single` 和 `test_frcnn_single` 脚本进行测试，命令如下：

```bash
python tools/test.py $config_file $weights_file --out $output_dir_file --work-dir=$output_dir
```
$config_file：训练模型output中的config文件

$weights_file：训练好的权重文件

$output_dir_file：测试输出的pkl文件

$output_dir：输出目录

如果需要记录验证集上的loss和mAP，请将config中的test数据改成vali，并且修改你的环境中mmengine包中的runner/loops.py文件

修改里面的TestLoop函数

参考如下，请根据你的模型灵活调整记录的loss类型，loss和mAP记录会保存到json文件里。

```bash
@LOOPS.register_module()
class TestLoop(BaseLoop):
    """Loop for test.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 testing. Defaults to
            False.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 fp16: bool = False):
        super().__init__(runner, dataloader)

        if isinstance(evaluator, dict) or isinstance(evaluator, list):
            self.evaluator = runner.build_evaluator(evaluator)  # type: ignore
        else:
            self.evaluator = evaluator  # type: ignore
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.evaluator.dataset_meta = self.dataloader.dataset.metainfo
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in evaluator, metric and '
                'visualizer will be None.',
                logger='current',
                level=logging.WARNING)
        self.fp16 = fp16

    def run(self) -> dict:
        """Launch test."""
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')
        self.runner.model.eval()
        loss_list = []
        loss_cls = []
        loss_conf = []
        loss_xy = []
        loss_wh = []

        # for frcnn
        # loss_rpn_cls = []
        # loss_rpn_bbox = []
        # loss_cls = []
        # acc = []
        # loss_bbox = []

        result = {}
        for idx, data_batch in enumerate(self.dataloader):
            outputs = self.run_iter(idx, data_batch)
            # print(outputs)
            loss_list.append(outputs['loss'].item())
            loss_cls.append(outputs['loss_cls'].item())
            loss_conf.append(outputs['loss_conf'].item())
            loss_xy.append(outputs['loss_xy'].item())
            loss_wh.append(outputs['loss_wh'].item())

            # loss_rpn_cls.append(outputs['loss_rpn_cls'].item())
            # loss_rpn_bbox.append(outputs['loss_rpn_bbox'].item())
            # loss_cls.append(outputs['loss_cls'].item())
            # acc.append(outputs['acc'].item())
            # loss_bbox.append(outputs['loss_bbox'].item())

        li_list = [loss_list, loss_cls, loss_conf, loss_xy, loss_wh]
        li_names = ["loss", "loss_cls", "loss_conf", "loss_xy", "loss_wh"]

        # li_list = [loss_list, loss_rpn_cls, loss_rpn_bbox, loss_cls, acc, loss_bbox]
        # li_names = ["loss", "loss_rpn_cls", "loss_rpn_bbox", "loss_cls", "acc", "loss_bbox"]
        for j in range(len(li_list)):
            li = li_list[j]
            total_loss = 0
            for i in range(len(li)):
                total_loss += li[i]
            result[li_names[j]] = total_loss / len(li)
        print(result)
        # compute metrics
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        # print(metrics)
        result["pascal_voc/mAP"] = metrics["pascal_voc/mAP"]
        result["pascal_voc/AP50"] = metrics["pascal_voc/AP50"]
        import json
        with open("yourpath/mmdetection/vali_loss_yolov3.jsonl", "a+", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False)+"\n")
        self.runner.call_hook('after_test_epoch', metrics=metrics)
        self.runner.call_hook('after_test')
        return metrics

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)
        # predictions should be sequence of BaseDataElement
        with autocast(enabled=self.fp16):
            outputs = self.runner.model.test_step(data_batch)
        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        

        def val_step(model, data):
            data = model.data_preprocessor(data, True)
            losses = model(**data, mode='loss')  # type: ignore
            parsed_losses, log_vars = model.parse_losses(losses)
            return log_vars
        model = self.runner.model
        outputs = val_step(model, data_batch)
        # print(outputs)

        self.runner.call_hook(
            'after_test_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
        return outputs
```

## 引用
```bibtex
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and


             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```
