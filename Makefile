.PHONY: venv run dry

venv:
	python3 -m venv .venv

run:
	python3 mbox_thread_dump.py --mbox "All mail Including Spam and Trash-002.mbox" --targets targets.txt --outdir out/ --out-mbox out/filtered.mbox

dry:
	python3 mbox_thread_dump.py --mbox "All mail Including Spam and Trash-002.mbox" --targets targets.txt --outdir out/ --out-mbox out/filtered.mbox --dry-run
