export CUDA_VISIBLE_DEVICES=5
for epoch in $(seq 10 10 100)
do
  model_path="./output_0528_yolo416/epoch_${epoch}.pth"
  python tools/test.py ./output_yolo416/yolov3_d53_8xb8-ms-416-273e_coco.py  $model_path --out "./output_test_wyq_yolov3/out_${epoch}.pkl" --work-dir="./output_test_wyq_yolov3"
done

