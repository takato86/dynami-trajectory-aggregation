for JSON in "linrs.json" "nrs.json"
do
echo "in/configs/${JSON}"
xvfb-run -a python main.py --config="in/configs/${JSON}"
done
