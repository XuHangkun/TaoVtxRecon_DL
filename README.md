## TAO Vertex Reconstruction Based on Deep Learning

### Experiments
- train the model
```shell
$ python -m torch.distributed.launch --nproc_per_node=num_of_gpus distributed_train.py --exp_name experiment_name --batch_size batch_size --model_name VGG --epoch 12
```

- evaluate the model
```shell
$ python evaluation.py --ckpt /path/to/pretrained/model --eval_output /path/to/output/file
```