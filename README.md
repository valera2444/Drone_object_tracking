Object tracking in videos from VisDrone2019 dataset. 
DeepSORT used as a tracking algorithm. Tested only with python3.11.9. 
Libraries used:
1) [torchreid](https://github.com/KaiyangZhou/deep-person-reid.git). Added as a folder
2) [DeepSORT](https://github.com/levan92/deep_sort_realtime.git). Added as a folder
3) [MOT metrics](https://github.com/cheind/py-motmetrics.git). Used via pip
4) [SORT](https://github.com/abewley/sort.git).Added as sort.py file

You may use run.sh as an example of running run.py file. For more info about it please run 
```
python3.11 run.py -h
```

docker_run.sh script provides an example of running docker container

For running container with GPU you must have nvidia-container-toolkit installed. Image size - 6.72GB(6.1GB - site-packages).

Please create example_for_docker_image folder that must contain (this used only by docker):
1) annotations.txt 
2) Folder with sequence of image


