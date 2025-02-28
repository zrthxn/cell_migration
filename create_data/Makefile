%.done: %.todo
	rm -rf $*
	mkdir $*
	cd $* && python3 ../startme.py 100
	touch $*.done
