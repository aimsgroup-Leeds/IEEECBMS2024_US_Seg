#$ -V -cwd
#$ -l h_rt=24:00:00
#$ -l h_vmem=12G
#$ -l coproc_v100=1
#$ -m be
#$ -N efficientNet_backbone


module load cuda
ROOTFOLDER=/resstore/b0211/Users/scee/MSC-US-Seg/

python ${ROOTFOLDER}main_US_Seg.py \
   --data_root '../../../Data'
   --model 'efficientnet-Unet'
   --backbone 'efficientnet_b2'
