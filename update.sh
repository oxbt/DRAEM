script_dir=$(cd $(dirname $0);pwd)
echo "change dir to $script_dir"
cd $script_dir

now=$(date "+%Y-%m-%d")
echo "Starting add-commit-pull-push..."

git add -A && git commit -m "$now" && git pull && git push origin main
echo "Finish!"
read
