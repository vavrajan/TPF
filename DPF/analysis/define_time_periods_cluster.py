## Import global packages
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from absl import app
from absl import flags


## Import local modules
# Necessary to import local modules from specific directory
import sys
# first directory here is the one where analysis_cluster is located
# this is ./DPF/analysis/analysis_cluster
# So add ./ to the list so that it can find ./DPF.code....
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))

from DPF.code.define_time_periods import define_time_periods

## FLAGS
flags.DEFINE_string("data_name", default="hein-daily", help="Data source being used.")
flags.DEFINE_string("addendum", default="", help="String to be added to data name."
                                                 "For example, for senate speeches the session number.")
flags.DEFINE_string("time_periods", default="sessions",
                    help="String that defines the set of dates that break the data into time-periods.")
flags.DEFINE_integer("min_word_count", default=1,
                    help="The minimal number of word usage to be included in the dataset.")
flags.DEFINE_boolean("min_word_count_per_time", default=False,
                     help="A boolean declaring whether the minimal word count"
                          "should be satisfied for each and every time period separately (True)"
                          "or for all time periods combined.")

FLAGS = flags.FLAGS

def main(argv):
    del argv

    ### Setting up directories
    project_dir = os.getcwd()
    source_dir = os.path.join(project_dir, 'data', FLAGS.data_name)
    data_dir = os.path.join(source_dir, 'clean')

    define_time_periods(FLAGS.data_name, data_dir, FLAGS.addendum, FLAGS.time_periods,
                        FLAGS.min_word_count, FLAGS.min_word_count_per_time)

if __name__ == '__main__':
    app.run(main)
