gpu=0
model=spaounet3dall
data=liver3Dnpy
crop=96
datadir=datasets
batch=2
iterations=100000
lr=1e-3
momentum=0.99
snapshot=10000
result=learned_models
loss_criterion=wbce

validdata=${data}
outdir=${result}/${data}-${model}-${loss_criterion}/${model}
mkdir -p ${result}/${data}-${model}-${loss_criterion}
python train.py ${outdir} --model ${model} \
    --gpu ${gpu} \
    --lr ${lr} -b ${batch} -m ${momentum} \
    --crop_size ${crop} --iterations ${iterations} \
    --snapshot ${snapshot} \
    --datadir ${datadir} \
    --dataset ${data} \
    --validdata ${validdata} \
    --loss_criterion ${loss_criterion} \
    --use_validation \
    --model3d
