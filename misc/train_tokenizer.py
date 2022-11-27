from tokenizers import Tokenizer

from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

from pathlib import Path

# PATH = '/home/hu/database/CUB_200_2011/text/'

# files = list(str(i) for i in Path(PATH).glob('**/*.txt'))

# tokenizer = Tokenizer(BPE(unk_token="[MASK]"))
# tokenizer.pre_tokenizer = Whitespace()
# trainer = BpeTrainer(vocab_size=512, special_tokens=["[PAD]","[MASK]"])

# tokenizer.train(files=files, trainer=trainer)

# tokenizer.save('/home/hu/UniDm/misc/cub_tokenizer.json')

tokenizer = Tokenizer.from_file(path="/home/hu/UniDm/misc/cub_tokenizer.json")
tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=64)
o = tokenizer.encode("I am boy")