Blotto:
	echo "#!/bin/bash" > Blotto
	echo "python3 main.py \"\$$@\"" >> Blotto
	chmod u+x Blotto
