##########
# Import #
##############################################################################

from space_time_modeling.eda import eda

#######
# Use #
##############################################################################
if __name__ == "__main__":
    
    #############
    # Attribute #
    ##########################################################################
    
    plots = [
        "data_date_trend", 
        "pair_plot", 
        "acf_plot", 
        "pacf_plot", 
        "rolling_statistics",
        "correlation_plot",
    ]
    
    plot_attribute = {
        "control_column": "Date",
        "target_column": "Open",
    }
    
    ###########
    # Get eda #
    ##########################################################################
    
    eda(
        df = r"result/test.csv",
        store_at = "result/eda_fed",
        plot = plots,
        plot_attribute = plot_attribute
    )
    
    ##########################################################################

##############################################################################
