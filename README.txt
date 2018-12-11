Instructions to run code:
To train model on a genre and generate weights:
python ./main.py train [genre]

To have the model generate lyrics on the genre it was trained on:
python ./main.py generate [genre] --seq [length of output]

Available genres are: Pop, Rock, Jazz, Hip-Hop, Metal and Country 