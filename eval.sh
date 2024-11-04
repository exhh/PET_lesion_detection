gpu=1
traindata=liver3Dnpy
testdata=${traindata}
model=spaounet3dall
loss_criterion=wbce
result=learned_models
datadir=datasets
eval_result_folder=experiments
split_train=train
split_val=val
split_test=test
eval_mode3d=3d
eval_mode2d=2d

for iteration in last
do
  if [ ${iteration} == 'last' ]
  then
    load_model=${result}/${traindata}-${model}-${loss_criterion}/${model}.pth
  else
    load_model=${result}/${traindata}-${model}-${loss_criterion}/${model}-${iteration}.pth
  fi
  predict_type=ensem
  python eval.py ${load_model} --model ${model} \
      --gpu ${gpu} \
      --datadir ${datadir} \
      --dataset ${testdata} \
      --eval_result_folder ${eval_result_folder} \
      --train_val_test ${split_test} \
      --eval2d_3d ${eval_mode3d} \
      --predict_type ${predict_type} \
      --eval_det
done
