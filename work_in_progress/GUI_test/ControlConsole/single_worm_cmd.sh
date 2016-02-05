MAX_NUM_PROCESS=8

for SET_N in 15 14 13 10
do
python3 compressExpLocalChecked.py \
/Volumes/behavgenom_archive\$/thecus/nas207-3/Data/from\ pc207-$SET_N/ \
/Volumes/behavgenom_archive\$/MaskedVideos/nas207-3/Data/from\ pc207-$SET_N/ \
--pattern_include '*.avi' --pattern_exclude '*swimming*' --json \
/Users/ajaver/Desktop/JSON_singleworm/on_food.json --max_num_process $MAX_NUM_PROCESS
done

