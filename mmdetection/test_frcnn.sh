export CUDA_VISIBLE_DEVICES=5
for epoch in $(seq 2 2 24)
do
  model_path="./output_0528_frcnn2x/epoch_${epoch}.pth"
  python tools/test.py ./output_frcnn2x/faster-rcnn_r50_fpn_2x_coco.py  $model_path --out "./output_test_wyq_frcnn/out_${epoch}.pkl" --work-dir="./output_test_wyq_frcnn"
done
