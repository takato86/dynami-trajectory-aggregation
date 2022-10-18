for JSON in "linaive_human_1.ini" "linaive_human_2.ini" "linaive_human_3.ini" "linaive_human_4.ini" "linaive_human_5.ini" "naive_human_1.ini" "naive_human_2.ini"
do
echo "in/configs/${JSON}"
xvfb-run -a python main.py --config="in/configs/${JSON}"
done
