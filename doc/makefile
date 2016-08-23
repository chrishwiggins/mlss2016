#################################
#
# makefile for talks
# chris.wiggins@gmail.com
#
#################################
#
# dependencies:
# 	pandoc (needed)
# 	qpdf (can be commented out, just for reporting)
# 	gs (for making wiggins-full from PDF surgery)
#
#
# input files needed for wiggins.pdf:
# 	wiggins-raw.md
#
#
# figure directory:
# 	fig
#
#
# other files needed
# 	makefile: this file
# 	mlss2016-nyt.pdf: nontech talk deck, needed only for surgery/"full"
#
#
# outputs:
# 	wiggins-full.pdf: complete, surgically created
# 	wiggins.pdf
# 	bdraft.pdf: draft in beamer style but w/o images 
# 		(output from "make bdraft")
# 	draft.pdf: draft, as a document 
# 		(output from "make draft")
# 	outline.pdf: outline 
# 		(output from "make out")
#   wiggins-draft.md (aux markdown file for "make draft")
#
#
# aux directory:
# 	backups (full of backups of edited file)
#
#################################

# variables:
 
ofiles =  \
 		wiggins-full.pdf \
		wiggins-1.pdf \
		wiggins-3.pdf \
 		wiggins-draft.md \
 		wiggins.pdf \
 		bdraft.pdf \
 		draft.pdf \
 		outline.pdf \
		wiggins.md 

# default make target:
 
all:
	make wiggins.pdf

# all targets:
 
everything:
	make draft
	make outline
	make bdraft
	make all
	make full

bdraft: 
	make bdraft.pdf

bdraft.pdf: wiggins-draft.md 
	time pandoc wiggins-draft.md -o bdraft.pdf -t beamer

draft: 
	make draft.pdf

draft.pdf: wiggins-draft.md 
	time pandoc wiggins-draft.md -o draft.pdf 

outline: 
	make outline.pdf

outline.pdf: wiggins.md 
	pandoc --toc --number-sections wiggins.md -o outline.pdf

# misc dependent files
 
wiggins-draft.md: wiggins.md
	grep -v '^\!\[' wiggins.md > wiggins-draft.md

wiggins.md: wiggins-raw.md 
	touch wiggins.md
	chmod +w wiggins.md
	# grep -v '%%%' wiggins-raw.md | sed -f wiggins.sed > wiggins.md
	grep -v '%%%' wiggins-raw.md > wiggins.md
	chmod -w wiggins.md

wiggins.pdf: wiggins.md 
	time pandoc --slide-level 2 -i -t beamer wiggins.md -o wiggins.pdf
	qpdf --show-npages wiggins.pdf

# 'full' (3 files stitched together)
 
wiggins-1.pdf: wiggins.pdf
	yes | gs -dBATCH -sOutputFile=wiggins-1.pdf -dFirstPage=1 -dLastPage=8 -sDEVICE=pdfwrite wiggins.pdf > /dev/null

wiggins-3.pdf: wiggins.pdf
	yes | gs -dBATCH -sOutputFile=wiggins-3.pdf -dFirstPage=9 -dLastPage=299 -sDEVICE=pdfwrite wiggins.pdf > /dev/null

full: 
	make wiggins-full.pdf

wiggins-full.pdf: wiggins-1.pdf mlss2016-nyt.pdf wiggins-3.pdf
	gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -sOutputFile=wiggins-full.pdf wiggins-1.pdf mlss2016-nyt.pdf wiggins-3.pdf
	# rm wiggins-1.pdf wiggins-3.pdf

## no dependences, always do:

remake: 
	vi makefile

edit:
	vi wiggins-raw.md
	cp wiggins-raw.md backups/wiggins-raw-`date +%Y-%m-%dT%H:%M:%S`.md

out:
	open outline.pdf

open:
	killall Preview
	open wiggins.pdf

test:
	date
	make clean
	make everything
	# should say "up to date"
	make everything
	date

clean:
	rm -f ${ofiles}

move:
	mv -f ${ofiles} ../out

spell:
	aspell -c wiggins-raw.md
