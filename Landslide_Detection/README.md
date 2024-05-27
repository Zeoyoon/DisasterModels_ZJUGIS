# Mask R-CNN for Landslide Detection and Segmentation

### model
仅使用遥感影像的滑坡提取模型：
model/mask_rcnn_landslide_0020.h5

结合致灾因素的滑坡提取模型：
model/mask_rcnn_landslide_0032.h5

打包的时候可以分开打包，避免压缩包过大

### sample
images/landslides 包含测试样本18张
images/landslides/dem 包含测试样本对应的高程数据，文件名称与遥感影像样本保持一致
images/landslides/slope 包含测试样本对应的坡度数据
images/landslides/water 包含测试样本对应的水系数据

### run
仅使用遥感影像的滑坡提取：DetectLandslide.py
运行命令

```
python run -u DetectLandslide.py \
--image=images/landslides/1024_area2_x5_y13.jpg \
--result=images/result
```

结合致灾因素的滑坡提取：DetectLandslideV2.py
运行命令
```
python run -u DetectLandslideV2.py \
--image=images/landslides/1024_area2_x5_y13.jpg \
--result=images/result_v2 \
--dem=images/landslides/dem/1024_area2_x5_y13.jpg \
--slope=images/landslides/slope/1024_area2_x5_y13.jpg \
--river=images/landslides/water/1024_area2_x5_y13.jpg
```

注意，水系参数名叫river但是文件夹名字叫water
