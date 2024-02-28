ckpt=`find . -name "*.ckpt" | xargs ls -lt | awk '{ print $NF }' | grep -v '^$' | head -n 1`
echo $ckpt
python validate/validate_cvpr.py --model=$ckpt --polos --foil --pascal --coef --flickr