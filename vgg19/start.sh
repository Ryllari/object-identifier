echo "*** Preparando para baixar depedencias ***\n"

git clone https://github.com/EscVM/OIDv4_ToolKit.git

pip install -r OIDv4_ToolKit/requirements.txt

python OIDv4_ToolKit/main.py downloader --classes classes.txt --type_csv train --limit 5000 --image_IsInside 0 --image_IsOccluded 0

