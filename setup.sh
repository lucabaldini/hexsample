#!/bin/bash

# See this stackoverflow question
# http://stackoverflow.com/questions/59895/getting-the-source-directory-of-a-bash-script-from-within
# for the magic in this command
SETUP_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

#
# Base package root. All the other releavant folders are relative to this
# location.
#
export HEXSAMPLE_ROOT=$SETUP_DIR

#
# Add the root folder to the $PYTHONPATH so that we can effectively import
# the relevant modules.
#
export PYTHONPATH=$HEXSAMPLE_ROOT:$PYTHONPATH


#
# Add the bin folder to the $PATH so that we have the executables off hand.
#
export PATH=$HEXSAMPLE_ROOT/baldaquin/bin:$PATH
