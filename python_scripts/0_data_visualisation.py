import matplotlib.pyplot as plt

# Document dates
years_ipcc = [1990, 1995, 2001, 2007, 2014, 2022]
years_ccm = [2005, 2008, 2014, 2022]

# Document names
ipcc_labels = ['AR1', 'AR2', 'AR3', 'AR4', 'AR5', 'AR6',]
ccm_labels = ['CCM 2005', 'CCM 2008', 'CCM 2014', 'CCM 2022']

# Values
ipcc_val = [1, 2, 3, 4, 5, 6]
ccm_val = [1, 2, 3, 4]


# Plot size
plt.figure(figsize=(8, 5))

# Style
plt.plot(years_ipcc, ipcc_val, marker='o', color='blue', label='IPCC WG3 SPM')
plt.plot(years_ccm, ccm_val, marker='o', color='purple', label='Wikipedia CCM')

# Text label positions are set individually by hand otherwise it overlaps
# IPCC
plt.annotate(ipcc_labels[0], (years_ipcc[0], ipcc_val[0]), xytext=(0, 0), textcoords='offset points', fontsize=9, color='blue')
plt.annotate(ipcc_labels[1], (years_ipcc[1], ipcc_val[1]), xytext=(0, 0), textcoords='offset points', fontsize=9, color='blue')
plt.annotate(ipcc_labels[2], (years_ipcc[2], ipcc_val[2]), xytext=(0, 0), textcoords='offset points', fontsize=9, color='blue')
plt.annotate(ipcc_labels[3], (years_ipcc[3], ipcc_val[3]), xytext=(0, 0), textcoords='offset points', fontsize=9, color='blue')
plt.annotate(ipcc_labels[4], (years_ipcc[4], ipcc_val[4]), xytext=(0, 0), textcoords='offset points', fontsize=9, color='blue')
plt.annotate(ipcc_labels[5], (years_ipcc[5], ipcc_val[5]), xytext=(0, 0), textcoords='offset points', fontsize=9, color='blue')
# Wiki
plt.annotate(ccm_labels[0], (years_ccm[0], ccm_val[0]), xytext=(0, 0), textcoords='offset points', fontsize=9, color='purple')
plt.annotate(ccm_labels[1], (years_ccm[1], ccm_val[1]), xytext=(0, 0), textcoords='offset points', fontsize=9, color='purple')
plt.annotate(ccm_labels[2], (years_ccm[2], ccm_val[2]), xytext=(0, 0), textcoords='offset points', fontsize=9, color='purple')
plt.annotate(ccm_labels[3], (years_ccm[3], ccm_val[3]), xytext=(0, 0), textcoords='offset points', fontsize=9, color='purple')

# Title
plt.xlabel('Year')
plt.ylabel("Current Metric")
plt.title("Title of Metric + Over Time")
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
plt.tight_layout() 
plt.show()