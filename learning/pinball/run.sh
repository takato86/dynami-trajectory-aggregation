for JSON in "dta.json" "linrs.json" "nrs.json" "srs.json"
do
echo "in/configs/${JSON}"
xvfb-run -a python main.py --config="in/configs/${JSON}"
done
