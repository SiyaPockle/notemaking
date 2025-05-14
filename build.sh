#!/bin/bash
set -e

echo "Installing build dependencies..."
sudo apt-get update
sudo apt-get install -y build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
  libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev \
  liblzma-dev python-openssl

# Get current Python version
PY_VERSION=$(python --version | cut -d' ' -f2)
echo "Current Python version: $PY_VERSION"

# Download and build SQLite
echo "Downloading and building newer SQLite..."
mkdir -p ~/sqlite_build
cd ~/sqlite_build
wget https://www.sqlite.org/2023/sqlite-autoconf-3410200.tar.gz
tar xzf sqlite-autoconf-3410200.tar.gz
cd sqlite-autoconf-3410200
./configure
make
sudo make install

# Update dynamic linker
sudo ldconfig

# Download and compile Python with the new SQLite
echo "Downloading and compiling Python $PY_VERSION..."
cd ~
wget https://www.python.org/ftp/python/$PY_VERSION/Python-$PY_VERSION.tgz
tar xzf Python-$PY_VERSION.tgz
cd Python-$PY_VERSION
./configure --enable-optimizations --with-system-ffi --with-computed-gotos --enable-loadable-sqlite-extensions
make -j $(nproc)
sudo make altinstall

echo "Creating symlinks..."
PY_SHORT_VERSION=$(echo $PY_VERSION | cut -d. -f1-2)
sudo ln -sf /usr/local/bin/python$PY_SHORT_VERSION /usr/local/bin/python
sudo ln -sf /usr/local/bin/pip$PY_SHORT_VERSION /usr/local/bin/pip

echo "Rebuilding your virtual environment..."
cd /workspaces/notemakinggg
rm -rf myenv
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt  # If you have a requirements file

echo "Checking SQLite version..."
python -c "import sqlite3; print(sqlite3.sqlite_version)"

echo "Done! Python has been rebuilt with newer SQLite."