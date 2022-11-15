for INI in "dynamic_human_0.1.ini" "dynamic_human_0.01.ini" "dynamic_human_0.001.ini"
do
INI_PATH="in/configs/step_env/${INI}"
echo $INI_PATH
python main.py --config=$INI_PATH
done
