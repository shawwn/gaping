set -ex
exec python3 -i -m pdb -c continue -m gaping.biggan_train --gin_config gaping/config/biggan256.gin --gin_bindings 'options.bar = 99' "$@"

