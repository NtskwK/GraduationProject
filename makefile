
# 安装环境
install_env:
	@conda env create -f environmental.yml -p .conda
	@pip install -r requirements.txt
# 导出环境
export_env:
	@conda env export > environmental.yml
	@pip freeze > requirements.txt

create_record:
	@python record/new.py