# %% [markdown]
# # **PROJECT 2: AN ANALYSIS OF MOBILE SALES**

# %% [markdown]
# ## 1. Import Necessary Libraries

# %%
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')
import plotly.io as pio
pio.renderers.default = "notebook_connected"

# %% [markdown]
# ## 2. Load Dataset

# %%
df = pd.read_csv("/Users/apple/Downloads/Mobiles_Dataset.csv")
df.head(30)

# %% [markdown]
# ## 3. Explore Dataset

# %%
df.info()

# %%
df.isnull().sum()

# %% [markdown]
# ## 4. Data Cleaning

# %%
#Extract Brand Name from Product Name
df['Brand'] = df['Product Name'].str.extract(r'^(\w+)')
df['Brand'] = df['Brand'].astype(str).apply(lambda x: x.title())
df['Product Name'] = df['Product Name'].astype(str).apply(lambda x: x.title())

#Remove the money symbol in all rows of columns Actual price and Discount price
df['Actual price ₹'] = df['Actual price'].str.replace('[₹,]','', regex = True)
df['Discount price ₹'] = df['Discount price'].str.replace('[₹,]','', regex = True)
df = df.drop(columns=['Actual price','Discount price'])

#Handle NIL values in Actual and Discount price columns
df['Actual price ₹'] = pd.to_numeric(df['Actual price ₹'], errors ='coerce')
df['Actual price ₹'] = df['Actual price ₹'].replace('NIL', df['Actual price ₹'].mean())
df['Actual price ₹'] = df['Actual price ₹'].fillna(df['Actual price ₹'].mean())
df['Discount price ₹'] = pd.to_numeric(df['Discount price ₹'], errors ='coerce')
df['Discount price ₹'] = df['Discount price ₹'].replace('NIL', df['Discount price ₹'].mean())
df['Discount price ₹'] = df['Discount price ₹'].fillna(df['Discount price ₹'].mean())

#Create Discount amount (%)
df['Discount amount (%)'] = round((df['Actual price ₹'] - df['Discount price ₹'])/df['Actual price ₹']*100,2)

#Remove Ratings and Reviews in two columns Ratings and Reviews
df['Number of Rating'] = df['Rating'].str.replace('[Ratings,]','', regex = True)
df['Number of Reviews'] = df['Reviews'].str.replace('[Reviews,]','', regex = True)
df = df.drop(columns=['Rating','Reviews'])

#Handle NIL values in the RAM (GB) (based on information of Description)
def extract_ram(description):
    extract_ram = re.search(r'(\d+)\s*(GB|MB)\s*RAM', description)
    ram = int(extract_ram.group(1)) if extract_ram else None
    return ram
df['RAM (GB)'] = df['Description'].apply(extract_ram)

#Handle NIL values in the Storage (GB) (based on information of Description)
def extract_storage(description):
    extract_storage = re.search(r'(\d+)\s*(GB|MB)\s*(?:ROM|Internal|Storage)', description)
    storage = int(extract_storage.group(1)) if extract_storage else None
    return storage
df['Storage (GB)'] = df['Description'].apply(extract_storage)

#Replace | to + in the Camera column
df['Camera'] = df['Camera'].str.replace('|','+')
#Handle Null values in the Camera column
df['Camera'] = df['Camera'].apply(lambda x: 'Not Present' if pd.isna(x) or x =='' else x)
df['Camera'] = df['Camera'].str.replace('0MP + 0MP','Not Present')

#Create Star Category and Price Category columns
df['Star Category'] = pd.cut(df['Stars'], bins = [0,3.4, 3.8, 4.2, 4.6, 5], labels =['Poor', 'Not Preferred', 'Fair', 'Good', 'Excellent'])
df['Price Category'] = pd.cut(df['Actual price ₹'], bins = [0, 10000, 20000, 30000, 40000, df['Actual price ₹'].max()], labels =['Low', 'Mid', 'High', 'Premium', 'Luxury'])

#Extract Main and Second Cameras
def extract_main_cam(camera):
    extract_main_camera = re.search(r'(\d+)MP\s*(\+)?\s*(\d+MP)?', camera)
    main_camera = int(extract_main_camera.group(1)) if extract_main_camera and camera != 'Not Present' else np.nan
    return main_camera
df['Main Camera'] = df['Camera'].apply(extract_main_cam)

def extract_second_cam(camera):
    extract_second_camera = re.search(r'(\d+)MP\s*(\+)?\s*(\d+)(MP)', camera)
    second_camera = int(extract_second_camera.group(3)) if extract_second_camera and camera != 'Not Present' else np.nan
    return second_camera
df['Second Camera'] = df['Camera'].apply(extract_second_cam)

#Define the desired column order
desired_order = ['Product Name', 'Brand', 'Price Category', 'Actual price ₹', 'Discount price ₹', 'Discount amount (%)', 'Stars', 'Star Category', 'Number of Rating', 'Number of Reviews', 'RAM (GB)', 'Storage (GB)', 'Display Size (inch)', 'Camera', 'Main Camera', 'Second Camera', 'Description', 'Link']
df = df[desired_order]

#Drop the Description and Link columns
df = df.drop(columns =['Description', 'Link'])

#Duplicate the original data for further price analysis (still contain Apple and other brands)
df_dup = df.copy()

#Handle the remaining null values in columns
print(df.isnull().sum())
df = df.dropna()

df_dup.head(10)


# %%
df['RAM (GB)'].unique()
df['Main Camera'].value_counts()
#Drop unreal RAM and Main Camera columns
df = df[df['RAM (GB)'] != 46875]
df_dup = df_dup[df_dup['RAM (GB)'] != 46875]
df = df[df['Main Camera'] != 108.0]
df_dup = df_dup[df_dup['Main Camera'] != 108.0]
#Print cleaned dataser
df.head(10)

# %%
df.info()

# %%
#Convert to category and int types in the cleaned dataset
df = df.astype({'Product Name': 'category', 'Brand':'category', 'Actual price ₹':'int',
                 'Discount price ₹': 'int', 'Number of Rating': 'int','Number of Reviews':'int', 'RAM (GB)': 'int', 'Storage (GB)': 'int', 
                 'Display Size (inch)': 'int', 'Main Camera': 'int', 'Second Camera': 'int'})
print(df.info())
print(df.describe(include= "all"))

# %%
df_dup.isnull().sum()

# %%
#Convert to category and int types in the not cleaned dataset
df_dup = df_dup.astype({'Product Name': 'category', 'Brand':'category', 'Actual price ₹':'int',
                 'Discount price ₹': 'int', 'Number of Rating': 'int','Number of Reviews':'int', 
                 'Display Size (inch)': 'int'})
df_dup.info()

# %% [markdown]
# ## 5. Data Visualization

# %% [markdown]
# ### 5.1 Overview

# %%
# Create histograms for each numeric column
numeric_df = df_dup.select_dtypes(include =[float,int])
numeric_df = pd.DataFrame(numeric_df)
print(numeric_df)
numeric_df_columns = numeric_df.columns
for column in numeric_df.columns:
    fig = px.histogram(df_dup, x=column, title=f'Distribution of {column}', 
                       nbins=10, marginal='box')
    fig.update_layout(font_color="grey", font_size =12,
        title_font_color="black", title_font_size =24)
    fig.show()

# %% [markdown]
# - Price Distribution:
#    + The actual price chart shows that most mobile phones are clustered around the ₹20,000 to ₹40,000 range, with fewer phones in the premium (₹40,000–₹80,000) and luxury (above ₹80,000) ranges
#    + Discount price distribution is heavily skewed toward lower prices, indicating aggressive discounting for mid-tier phones
# - Discount Distribution:
#    + Brands like Honor, Micromax, and Poco offer the highest discount percentages (up to 50%), while premium brands like Apple, Google, and Samsung offer minimal discounts
#    + Mid-range phones see the highest average discount rates (around 25%), with high-end models offering much smaller discounts, or in some cases, none at all
# - Star Distribtion:
#    + The majority of phones have star ratings between 4.2 and 4.4, with a small number achieving ratings above 4.5. Very few phones have ratings below 4.0, which suggests that most products are perceived as having decent quality by customers
#    + Brands with high ratings (4.5 and above) include premium players like Apple, OnePlus, and Samsung, where consumer satisfaction tends to be higher
#    + Brands with lower ratings (around 3.6 to 3.8) include lesser-known or budget brands like Vox, Karbonn, and Jio, indicating some level of dissatisfaction or unmet customer expectations
# - Review Distribution:
#    + Total reviews are heavily skewed towards well-known brands, with companies like Apple, Samsung, and Realme collecting the most reviews. Apple, for instance, has over 276,000 reviews, indicating a high level of customer engagement
#    + Brands with fewer reviews include Vox, Karbonn, and Jio, which have under 100 reviews, reflecting their limited market reach or consumer engagement
#    + Average reviews per product vary significantly, with premium brands typically garnering more reviews per product (e.g., Apple averages around 7,086 reviews per product) compared to budget brands (e.g., Vox with around 6 reviews per product)
# - RAM Distribution:
#    + The most common RAM configuration is 8GB, dominating the market with a significant share (around 361 entries). This is followed by 4GB and 12GB, with smaller shares
#    + High-end configurations like 16GB or 32GB RAM are relatively rare and are typically found in premium devices
#    + Low-end configurations like 2GB or 4GB RAM appear mostly in budget smartphones
# - Storage Distribution:
#    + 128GB storage is the most popular configuration, capturing a significant portion of the market (421 entries). This is followed by 256GB, with higher configurations like 512GB being less common and reserved for premium models
#    + Smaller configurations like 32GB and 64GB are seen in lower-end devices, while 4GB storage is very rare and usually found in ultra-budget or legacy models
# - Main Cam Distribution:
#    + The most common main camera configuration is 50MP, particularly in mid-range and high-end devices. The rest of the configurations include 48MP, 12MP, and a few lower-end models featuring 2MP cameras
#    + Higher-end models feature cameras in the range of 50MP, while budget phones stick to lower resolutions like 12MP or 2MP
# - Second Cam Distribution:
#    + The most common second camera resolution is 2MP, especially in mid-range phones with dual-camera setups. The higher-end models have 12MP or better second cameras
#    + Lower-tier models either don’t have second cameras or feature basic 2MP cameras as a secondary sensor 
# 

# %%
#Calculate star category of price category
star_price = df_dup.groupby('Price Category')['Stars'].mean().round(1).sort_values().reset_index()
star_price['Star Category'] = pd.cut(star_price['Stars'], bins=[0, 3.4, 3.8, 4.2, 4.6, 5], labels=['Poor', 'Not Preferred', 'Fair', 'Good', 'Excellent'])
print(star_price)

#Calculate price category of star category
price_star = df_dup.groupby('Star Category')['Actual price ₹'].mean().astype(int).sort_values().reset_index()
price_star['Price Category'] = pd.cut(price_star['Actual price ₹'], bins = [0, 10000, 20000, 30000, 40000, df_dup['Actual price ₹'].max()], labels =['Low', 'Mid', 'High', 'Premium', 'Luxury'])
print(price_star)

categoryarray=[0,4.1,4.2,4.3,4.5]

#Visualize star category of price category
fig= make_subplots(rows=1, cols=2, subplot_titles=('Star Category of Price Category', 'Price Category of Star Category'))

fig.add_bar(x=['Low', 'Mid'], y=[4.1, 4.2], marker_color='cornflowerblue', name='Fair', row=1, col=1)
fig.add_bar(x=['High', 'Premium','Luxury'], y=[4.3, 4.3,4.5], marker_color='royalblue', name='Good', row=1, col=1)
fig.update_xaxes(title_text='Price Category', type='category', row=1, col=1)
fig.update_yaxes(title_text='Stars', type='category', categoryorder='array', categoryarray=categoryarray, row=1, col=1)

#Visualize price category of star category
#To see for example, if the rate is good, then customers are willing to pay which price category
fig.add_bar(x=['Poor', 'Not Preferred'], y=[1699, 7746], marker_color='yellowgreen', name='Low', row=1, col=2)
fig.add_bar(x=['Fair'], y=[16466], marker_color='olivedrab', name='Mid', row=1, col=2)
fig.add_bar(x=['Excellent', 'Good'], y=[33631,34918], marker_color='darkolivegreen', name='Premium',row=1, col=2)
fig.update_xaxes(title_text='Star Category', type='category', row=1, col=2)
fig.update_yaxes(title_text='Price', type='category', categoryorder='array', categoryarray=df['Actual price ₹'], row=1, col=2)

fig.update_layout(title='Price Category vs Star Category',title_font_size=24, title_font_color='black', font_size=12, font_color='grey', height=600)
fig.show()

# %% [markdown]
# - Premium and Luxury Brands (first chart) tend to receive the highest ratings (above 4.3 stars), which shows that customers perceive these products as superior in quality, even though they are priced higher
# - Low and Mid-range Price Categories (first chart) have comparatively lower star ratings, reflecting that price-sensitive customers might be more critical of product performance or experience
# - The second chart shows that as star ratings increase, the price of the products also rises. Products with ratings categorized as "Excellent" and "Good" fall mostly into the higher price range (Premium category). This mean that if customers feel 'good', they are willing to pay high amount of money for a new phone.

# %%
#Visualize top 10 most common Display Size (inch)
most_common_display = df_dup['Display Size (inch)'].value_counts().reset_index()
most_common_display =most_common_display.rename(columns={'count':'Number of Display Size (inch)'})
most_common_display['Common Display Size Percentage']= (most_common_display['Number of Display Size (inch)']/len(df['Display Size (inch)'])*100).round(2)
most_common_display = pd.DataFrame(most_common_display)
most_common_display = most_common_display[most_common_display['Display Size (inch)']!=0]
print(most_common_display)
display_pr_category = df_dup.groupby('Price Category')['Display Size (inch)'].mean().astype(int).reset_index()
print(display_pr_category)

fig1 = px.pie(most_common_display,values='Common Display Size Percentage', names='Display Size (inch)', 
             template="plotly_white", color_discrete_sequence=px.colors.qualitative.Vivid)
fig1.update_layout(title_font_size =24, title_text ='Percentage of Display Size (inch)', title_font_color="black", 
                  font_size=12,font_color="grey", height=500)

fig1.show()
#Visualize Average Display size of each Price Category
fig2 = px.bar(display_pr_category, x= 'Price Category', y='Display Size (inch)', title = 'Average Display Size (by Price Category)',
             color = 'Price Category', template="plotly_white", color_discrete_sequence=px.colors.qualitative.Vivid,
             text = 'Display Size (inch)')
fig2.update_traces(textposition='outside', texttemplate='%{text:.2s}')
fig2.update_layout(height=500, 
        font_color="grey", font_size =12,
        title_font_color="black", title_font_size =24)
fig2.show()


# %% [markdown]
# - At the present, the most common display size is 6 inch, followed by 1 and 2 inch. Meanwhile, the percentages of 7 and 5 inch phone are very small, illustrating that customers are leaning towards a 6 inch phone more than a very big one or a very small one.
# - The mean display size is also consistent with price categories, in which customers choosing mid, premium, and luxury brands usally select 6.0 inch.

# %% [markdown]
# ### 5.2 Brand Analysis

# %%
#Calculate top 10 most common brand
most_common_brand = df_dup['Brand'].value_counts().sort_values(ascending = False).reset_index()
most_common_brand = most_common_brand.rename(columns={'count':'Most Common Brand Count','Brand':'Common Brand'})
most_common_brand = most_common_brand.head(10)
#Calculate top 10 least common brand
least_common_brand = df_dup['Brand'].value_counts().sort_values(ascending = True).reset_index()
least_common_brand = least_common_brand.rename(columns={'count':'Least Common Brand Count','Brand':'Not Common Brand'})
least_common_brand = least_common_brand.head(10)
#Turn into dataframe
most_least_common_brand = pd.concat([most_common_brand, least_common_brand], axis=1)
print(most_least_common_brand)

#Visualise by drawing 2 bar charts side by side to compare
fig = make_subplots(rows=1, cols=2, subplot_titles = ('10 Most Common Brands', '10 Least Common Brands'))
fig.add_bar(x=most_least_common_brand['Common Brand'], y=most_least_common_brand['Most Common Brand Count'], name='Most Common', marker_color='lightseagreen', row =1, col=1, text = most_least_common_brand['Most Common Brand Count'])
fig.add_bar(x=most_least_common_brand['Not Common Brand'], y=most_least_common_brand['Least Common Brand Count'], name ='Least Common', marker_color='olivedrab', row=1, col=2, text = most_least_common_brand['Least Common Brand Count'])

fig.update_traces(textposition='outside', texttemplate='%{text:.2s}', row=1, col=1)
fig.update_traces(textposition='outside', texttemplate='%{text:.1s}', row=1, col=2)
fig.update_xaxes(title_text ='Brand', row=1, col=1)
fig.update_xaxes(title_text ='Brand', row=1, col=2)
fig.update_yaxes(title_text ='Count', row=1, col=1)
fig.update_yaxes(title_text ='Count', row=1, col=2)
fig.update_layout(title='Most and Least Common Brands', font_color="grey", font_size =12, title_font_color="black", title_font_size =24, height=600)
fig.show()

# %% [markdown]
# - Most Common Brands:
#    + Realme leads the market with 110 counts, followed by Samsung (100), Redmi (92), and Vivo (91)
#    + The top brands, especially Realme and Samsung, dominate significantly, with counts close to or above 100
# - Least Common Brands:
#    + Honor, Jio, and Vox appear at the bottom with only 1-2 counts
#    + Xiaomi, despite being a recognized brand, is listed among the least common, potentially indicating a region-specific trend or particular time period

# %%
#Calculate average star reviews of each brand and turn it into category
brand_star = df_dup.groupby('Brand')['Stars'].mean().round(1).sort_values().reset_index()
brand_star['Star Category'] = pd.cut(brand_star['Stars'], bins=[0, 3.4, 3.8, 4.2, 4.6, 5], labels=['Poor', 'Not Preferred', 'Fair', 'Good', 'Excellent'])
print(brand_star)

not_preferred_brand = brand_star[brand_star['Star Category']=='Not Preferred']
fair_brand = brand_star[brand_star['Star Category']=='Fair']
good_brand = brand_star[brand_star['Star Category']=='Good']

#Visualize average star category of each brand
#To see which one is most rated
fig= go.Figure()
fig.add_bar(x=not_preferred_brand['Brand'], y=not_preferred_brand['Stars'], marker_color='darkgrey', name='Not Preferred')
fig.add_bar(x=fair_brand['Brand'], y=fair_brand['Stars'], marker_color='turquoise', name='Fair')
fig.add_bar(x=good_brand['Brand'], y=good_brand['Stars'], marker_color='lightseagreen', name='Good')

fig.update_xaxes(type='category')
fig.update_yaxes(type='category', categoryorder='array', categoryarray=[0, 3.6, 3.7, 3.8, 3.9, 4.0, 4.2, 4.3, 4.4, 4.6])
fig.update_layout(title='Star Category (by Brand)',title_font_size=24, title_font_color='black', font_size=12, font_color='grey', height=600)
fig.show()

# %% [markdown]
# - Highly Rated Brands:
#    + Apple has the highest star rating, reaching close to 4.6, followed by OnePlus and Nothing, which are also highly rated above 4.5
#    + Popular brands like Xiaomi, Vivo, and Samsung are also rated well, all above 4.2
# - Low Rated Brands:
#    + Vox, Karbonn, and Jio receive the lowest ratings, between 3.6 and 3.8, categorizing them as "Not Preferred."

# %%
#Calculate the mean of actual and discount price of each brand
average_actual_discount_price = df_dup.groupby('Brand')[['Actual price ₹', 'Discount price ₹']].mean().sort_values(by='Actual price ₹', ascending=False).astype(int)
average_actual_discount_price = pd.DataFrame(average_actual_discount_price).reset_index()
print(average_actual_discount_price.head(10))

#Visualize the correlation of Actual & Discount Price (by Brand)
fig1 = px.scatter(df_dup, x= 'Actual price ₹', y='Discount price ₹', title ='Correlation of Actual & Discount Price (by Brand)',
                  color ='Brand', color_discrete_sequence=px.colors.qualitative.Light24, 
                  trendline="ols", trendline_scope="overall", trendline_color_override="lightseagreen")
fig1.update_layout(height=500, template="plotly_white",
        font_color="grey", font_size =12,
        title_font_color="black", title_font_size =24)
fig1.show()

#Visualise the mean of actual and discount price of each brand
fig2 = go.Figure(data=go.Bar(x=average_actual_discount_price['Brand'], y=average_actual_discount_price['Actual price ₹'], marker_color='lightseagreen',name='Actual', text = average_actual_discount_price['Actual price ₹'] ))
fig2.add_bar(x=average_actual_discount_price['Brand'], y=average_actual_discount_price['Discount price ₹'], name ='Discount', marker_color='yellowgreen',text =average_actual_discount_price['Discount price ₹'])

fig2.update_traces(textposition='outside', texttemplate='%{text:.2s}')    
fig2.update_xaxes(title_text ='Brand')
fig2.update_yaxes(title_text ='Price')
fig2.update_layout(font_color="grey", font_size =12, 
                  title='Average Actual & Average Discount Price (by Brand)', title_font_color="black", title_font_size =24, height=600)
fig2.show()

# %% [markdown]
# - Correlation of Actual & Discount Price (by Brand):
#    + The chart shows a strong positive correlation between actual price and discount price. More expensive brands like Apple, Samsung, and OnePlus offer higher discounts in absolute terms, even though their percentage discounts may be relatively small
#    + Brands like Realme, Oppo, and Vivo in the mid-range and low-price segments offer lower absolute discounts, but these are more impactful due to their lower price points
# - Most Expensive Brands:
#    + Apple stands out with an average price of 78k, followed by Xiaomi (65k) and Google (64k)
#    + The more premium brands tend to maintain higher price points, such as Honor (52k) and OnePlus (48k)
# - Brands with Lower Prices:
#    + Brands like Karbonn, Kechaoda, and Blackzone offer much lower prices, often in the 1-2k range
#    + There’s a clear divide in pricing between high-end brands (Apple, Google) and budget brands (Karbonn, Itel)

# %%
#Calculate average discount % of brand and price category
average_discount_amount = df_dup.groupby('Brand')['Discount amount (%)'].mean().round(2).sort_values(ascending=False).reset_index()
average_discount_amount = pd.DataFrame(average_discount_amount)
print(average_discount_amount.head(10))

avg_discount_price_category = df_dup.groupby('Price Category')['Discount amount (%)'].mean().round(2).sort_values(ascending=False).reset_index()
avg_discount_price_category = pd.DataFrame(avg_discount_price_category)
print(avg_discount_price_category)

#Visualize Average Discount Amount (%) of each Brand
average_discount_amount['color'] = "darkgray"
average_discount_amount['color'][0] = "royalblue"
average_discount_amount['color'][26] = "crimson"

fig1 = px.bar(average_discount_amount, x= 'Brand', y='Discount amount (%)', title = 'Average Discount Amount (%) (by Brand)',
             color = 'color', template="plotly_white", color_discrete_sequence=average_discount_amount.color.unique(),
             text = 'Discount amount (%)')
fig1.update_traces(textposition='outside', texttemplate='%{text:.2s}')
fig1.update_layout(showlegend=False, 
        font_color="grey", font_size =10,
        title_font_color="black", title_font_size =24)
fig1.show()

#Visualize Average Discount Amount (%) of each Price Category
fig2 = px.bar(avg_discount_price_category, x= 'Price Category', y='Discount amount (%)', title = 'Average Discount Amount (%) (by Price Category)',
             color = 'Price Category', template="plotly_white", color_discrete_sequence=px.colors.qualitative.Vivid,
             text = 'Discount amount (%)')
fig2.update_traces(textposition='outside', texttemplate='%{text:.2s}')
fig2.update_layout(height=500, 
        font_color="grey", font_size =12,
        title_font_color="black", title_font_size =24)
fig2.show()

# %% [markdown]
# In the first chart:
# - Highest Discounts:
#    + Vox offers the highest discount at 51%, followed by Honor (46%) and "I" (44%)
#    + These brands are likely using heavy discounting to drive sales, which aligns with the previous data showing lower ratings and market share
# - Negative Discount (Price Increase):
#    + Interestingly, Samsung shows a negative discount (-22%), suggesting that their prices may have increased rather than decreased
#    + Google also shows a very low discount of 3.7%, which is uncommon for the high-end brand category
# 
# In the second chart:
# - Products in the mid price category usually are discounted more than the other price categories
# - Meanwhile, those in the high price category are negatively discounted, with 7.1%

# %%
#Compare Total and average number of reviews of each Brand
total_mean_brand_reviews = df_dup.groupby('Brand')['Number of Reviews'].agg(Total_Number_of_Reviews='sum', Average_Number_of_Reviews='mean').reset_index()
total_mean_brand_reviews = total_mean_brand_reviews.rename(columns={'Total_Number_of_Reviews':'Total Number of Reviews', 'Average_Number_of_Reviews':'Average Number of Reviews'})
total_mean_brand_reviews['Average Number of Reviews']= total_mean_brand_reviews['Average Number of Reviews'].astype(int)
print(total_mean_brand_reviews)

fig = make_subplots(rows=1, cols=2, subplot_titles = ('Total # Reviews', 'Average # Reviews'))
fig.add_bar(x=total_mean_brand_reviews['Brand'], y=total_mean_brand_reviews['Total Number of Reviews'], name='Total', marker_color ='lightseagreen', row =1, col=1, text = total_mean_brand_reviews['Total Number of Reviews'])
fig.add_bar(x=total_mean_brand_reviews['Brand'], y=total_mean_brand_reviews['Average Number of Reviews'], name ='Average', marker_color ='yellowgreen',  row=1, col=2, text =total_mean_brand_reviews['Average Number of Reviews'])

fig.update_traces(textfont_size=12, textposition='outside', texttemplate='%{text:.2s}')
fig.update_xaxes(title_text='Brand', row=1,col=1)
fig.update_xaxes(title_text='Brand', row=1,col=2)
fig.update_yaxes(title_text='Total number', row=1,col=1)
fig.update_yaxes(title_text='Average number', row=1,col=2)
fig.update_layout(title="Total & Average Review Number (by Brand)", title_font_size =24, title_font_color="black", 
                  font_size=10,font_color="grey",
                  xaxis_tickangle=90)

fig.show()

# %% [markdown]
# Brands like Apple, Vivo, Motorola, and Samsung receive more reviews than the other brands, showing that these brands are more common than the other ones.

# %% [markdown]
# ### 5.3 RAM and Storage Analysis

# %%
#Most Common RAM and Storage (GB)
common_ram= df_dup['RAM (GB)'].value_counts().reset_index().astype(int)
common_storage= df_dup['Storage (GB)'].value_counts().reset_index().astype(int)
print(common_ram)
print(common_storage)

highest_ram = common_ram[common_ram['RAM (GB)']==8]
other_ram = common_ram[common_ram['RAM (GB)']!=8]

highest_storage = common_storage[common_storage['Storage (GB)']==128]
other_storage = common_storage[common_storage['Storage (GB)']!=128]

#Visualise by drawing 2 bar charts side by side to compare
fig = make_subplots(rows=1, cols=2, subplot_titles = ('RAM Count', 'Storage Count'))
fig.add_bar(x=highest_ram['RAM (GB)'], y=highest_ram['count'], textposition='outside',
            marker_color='royalblue', text=highest_ram['count'], name='Most common RAM',row =1, col=1)
fig.add_bar(x=other_ram['RAM (GB)'], y=other_ram['count'], marker_color='lightsteelblue', name='Other RAM',row =1, col=1)
fig.add_bar(x=highest_storage['Storage (GB)'], y=highest_storage['count'], textposition='outside',
            marker_color='darkorange', text=highest_storage['count'], name='Most common storage',row =1, col=2)
fig.add_bar(x=other_storage['Storage (GB)'], y=other_ram['count'], marker_color='bisque', name='Other storage',row =1, col=2)

fig.update_xaxes(title_text ='RAM (GB)', type='category', row=1, col=1)
fig.update_xaxes(title_text ='Storage (GB)', type='category', row=1, col=2)
fig.update_yaxes(title_text ='Count', row=1, col=1)
fig.update_yaxes(title_text ='Count', row=1, col=2)
fig.update_layout(template='plotly_white', title='Most Common RAM & Storage', font_color="grey", font_size =12, title_font_color="black", title_font_size =24, height=600)
fig.show()
print('The most common RAM is 8GB, while he most common storage is 128GB')

# %% [markdown]
# - RAM Count Analysis:
#    + 8 GB RAM is the most common among the devices, with a count of 361, indicating it's a popular choice for many consumers.
#    + 4 GB RAM and 12 GB RAM are also significant but much less common than 8 GB.
#    + Larger RAM sizes like 32 GB and 64 GB are far less common, possibly indicating that they are either higher-end options or less in demand.
# 
# - Storage Count Analysis:  
#    + 128 GB storage is the most common, with a count of 421, suggesting it is the preferred option for many buyers.
#    + 256 GB and 64 GB storage options are also fairly popular but do not match the prevalence of 128 GB.
#    + Higher storage capacities like 512 GB are much less common, possibly due to higher price points or being niche products.

# %%
#Calculate average star reviews of each RAM and Storage
ram_star = df_dup.groupby('RAM (GB)')['Stars'].mean().round(1).sort_values().reset_index()
storage_star = df_dup.groupby('Storage (GB)')['Stars'].mean().round(1).sort_values().reset_index()
ram_star['Star Category'] = pd.cut(brand_star['Stars'], bins=[0, 3.4, 3.8, 4.2, 4.6, 5], labels=['Poor', 'Not Preferred', 'Fair', 'Good', 'Excellent'])
storage_star['Star Category'] = pd.cut(brand_star['Stars'], bins=[0, 3.4, 3.8, 4.2, 4.6, 5], labels=['Poor', 'Not Preferred', 'Fair', 'Good', 'Excellent'])
print(ram_star)
print(storage_star)

not_preferred_ram = ram_star[ram_star['Star Category']=='Not Preferred']
not_preferred_storage = storage_star[storage_star['Star Category']=='Not Preferred']
fair_ram = ram_star[brand_star['Star Category']=='Fair']
fair_storage = storage_star[brand_star['Star Category']=='Fair']


#Visualize average star category of RAM and Storage
#To see which one is most preferred
fig=make_subplots(rows=1,cols=2, subplot_titles=('Star Rate of RAM', 'Star Rate of Storage'))
fig.add_bar(x=not_preferred_ram['RAM (GB)'], y=not_preferred_ram['Stars'], marker_color='darkgrey', name='Not Preferred RAM',row=1,col=1)
fig.add_bar(x=fair_ram['RAM (GB)'], y=fair_ram['Stars'], marker_color='royalblue', name='Fair RAM', opacity=0.85,row=1,col=1)
fig.add_bar(x=not_preferred_storage['Storage (GB)'], y=not_preferred_storage['Stars'], marker_color='darkgrey', name='Not Preferred Storage', row=1,col=2)
fig.add_bar(x=fair_storage['Storage (GB)'], y=fair_storage['Stars'], marker_color='darkorange', name='Fair Storage',opacity=0.85, row=1,col=2)

fig.update_xaxes(title_text='RAM (GB)', type='category', row=1,col=1)
fig.update_yaxes(title_text='Stars', type='category', categoryorder='array', categoryarray=[0, 3.6, 3.8, 3.9, 4.0, 4.2, 4.3, 4.5],row=1,col=1)
fig.update_yaxes(title_text='Stars', type='category', categoryorder='array', categoryarray=[0, 3.6, 3.7, 3.9, 4.0, 4.1, 4.2, 4.3],row=1,col=2)
fig.update_xaxes(title_text='Storage (GB)', type='category',row=1,col=2)
fig.update_layout(title='Star Category of RAM & Storage',title_font_size=24, title_font_color='black', font_size=12, font_color='grey')
fig.show()

# %% [markdown]
# - Star Rate of RAM:
#    + Devices with 8 GB, 12 GB, and 16 GB RAM have higher average ratings (around 4.2 to 4.4 stars), suggesting that these configurations meet customer expectations better
#    + 2 GB, 20 GB, 48 GB, and 64 GB RAM options have lower ratings, possibly due to performance limitations or being less balanced for typical usage
# 
# - Star Rate of Storage:
#    + Storage sizes like 64 GB, 128 GB, 256 GB, and 512 GB tend to have higher ratings (above 4 stars).
#    + Smaller storage options like 3 GB, 4 GB, and 5 GB have lower ratings, likely due to limited capacity for modern app and media needs
# 

# %%
#Calculate actual and discount price of ram and storage
#To see if RAM and Storage affect price
ram_avgprice = df_dup.groupby('RAM (GB)')['Actual price ₹'].mean().sort_values(ascending=False).reset_index().astype(int)
ram_avg_dis_price = df_dup.groupby('RAM (GB)')['Discount price ₹'].mean().sort_values(ascending=False).reset_index().astype(int)
ram_avg_dis_price = ram_avg_dis_price.drop(columns='RAM (GB)')
ram_avg_price = pd.concat([ram_avgprice, ram_avg_dis_price], axis=1) 

storage_avgprice = df_dup.groupby('Storage (GB)')['Actual price ₹'].mean().sort_values(ascending=False).reset_index().astype(int)
storage_avgprice = storage_avgprice[storage_avgprice!=0]
storage_avg_dis_price = df_dup.groupby('Storage (GB)')['Discount price ₹'].mean().sort_values(ascending=False).reset_index().astype(int)
storage_avg_dis_price = storage_avg_dis_price.drop(columns='Storage (GB)')
storage_avg_dis_price = storage_avg_dis_price[storage_avg_dis_price!=0]
storage_avg_price = pd.concat([storage_avgprice, storage_avg_dis_price], axis=1) 

print(ram_avg_price)
print(storage_avg_price)

#Visualise by drawing 2 bar charts side by side to compare
fig = make_subplots(rows=1, cols=2, subplot_titles = ('Price of RAM', 'Price of Storage'))
fig.add_bar(x=ram_avg_price['RAM (GB)'], y=ram_avg_price['Actual price ₹'], name='Actual price of RAM', text = ram_avg_price['Actual price ₹'], marker_color ='royalblue', row =1, col=1)
fig.add_bar(x=ram_avg_price['RAM (GB)'], y=ram_avg_price['Discount price ₹'], name='Discount price of RAM',text = ram_avg_price['Discount price ₹'], marker_color ='cornflowerblue',row =1, col=1)
fig.add_bar(x=storage_avg_price['Storage (GB)'], y=storage_avg_price['Actual price ₹'], name = 'Actual price of Storage',text = storage_avg_price['Actual price ₹'],marker_color ='darkorange',row=1, col=2)
fig.add_bar(x=storage_avg_price['Storage (GB)'], y=storage_avg_price['Discount price ₹'], name = 'Discount price of Storage', text = storage_avg_price['Actual price ₹'], marker_color ='sandybrown',row=1, col=2)

fig.update_traces(textposition='outside', texttemplate='%{text:.2s}')
fig.update_xaxes(title_text ='RAM (GB)', type='category', tickfont=dict(size=12), row=1, col=1)
fig.update_xaxes(title_text ='Storage (GB)', type='category', tickfont=dict(size=12), row=1, col=2)
fig.update_yaxes(title_text ='Price', row=1, col=1)
fig.update_yaxes(title_text ='Price', row=1, col=2)
fig.update_layout(title='Average Actual & Discount price of RAM & Storage',
                  font_color="grey", font_size =10, title_font_color="black", title_font_size =24, height=600)
fig.show()

# %% [markdown]
# High RAM and Storage configurations (16GB RAM, 512GB storage) are mostly seen in high-end phones, while 2GB or 4GB RAM is seen in low-end phones.

# %%
#Calculate average discount % of ram and storage
ram_discount_amount = df_dup.groupby('RAM (GB)')['Discount amount (%)'].mean().round(2).sort_values(ascending=False).reset_index()
storage_discount_amount = df_dup.groupby('Storage (GB)')['Discount amount (%)'].mean().round(2).sort_values(ascending=False).reset_index()
storage_discount_amount = storage_discount_amount[storage_discount_amount['Storage (GB)']!=0]
print(ram_discount_amount)
print(storage_discount_amount)

#Visualize average discount % of ram and storage
fig = make_subplots(rows=1, cols=2, subplot_titles=('RAM Average Discount %','Storage Average Discount %'))
fig.add_bar(x=ram_discount_amount['RAM (GB)'], y=ram_discount_amount['Discount amount (%)'], name='RAM', text = ram_discount_amount['Discount amount (%)'], marker_color ='royalblue', row=1, col=1)
fig.add_bar(x=storage_discount_amount['Storage (GB)'], y=storage_discount_amount['Discount amount (%)'], name ='Storage', text=storage_discount_amount['Discount amount (%)'], marker_color='darkorange', row=1, col=2)

fig.update_traces(textposition='outside', texttemplate='%{text:.2f%}')
fig.update_xaxes(title_text ='RAM (GB)', type='category', row=1, col=1)
fig.update_xaxes(title_text ='Storage (GB)', type='category', row=1, col=2)
fig.update_yaxes(title_text ='Discount Amount', row=1, col=1)
fig.update_yaxes(title_text ='Discount Amount', row=1, col=1)
fig.update_layout(font_color="grey", font_size =10, title='Average Discount Amount of Ram & Storage', title_font_color="black", title_font_size =24, height=600)
fig.show()


# %% [markdown]
# - Regarding RAM, while 48 and 20 GB RAM have high discount amount to boost sales, the opposite is seen in 12 GB RAM
# - Regarding Storage, while the discount amount trend for 48 and 20 GB Storage is the same as RAM, that of 512 GB reiceives negative discount

# %%
#Visualize scatter plot of Price with RAM, Storage, Main, Second Cam
selected_columns = ['RAM (GB)', 'Storage (GB)', 'Main Camera', 'Second Camera']
selected_columns1 = ['RAM (GB)', 'Storage (GB)', 'Main Camera', 'Second Camera', 'Actual price ₹']
rows, cols = 2, 2
fig = make_subplots(rows=rows, cols=cols, 
                    subplot_titles=('Ram and Actual Price ₹', 'Storage and Actual Price ₹',  'Main Camera and Actual Price ₹',  'Second Camera and Actual Price ₹'),
                    vertical_spacing=0.2)

for i, column in enumerate(selected_columns):
    row = i // cols + 1  
    col = i % cols + 1   
    fig.add_scatter(x=df[column], y=df['Actual price ₹'], mode='markers', showlegend=False, row=row, col=col)
    fig.update_xaxes(title_text=column, row=row, col=col)
    fig.update_yaxes(title_text='Price', row=row)

fig.update_traces(marker_color= 'royalblue', row=1,col=1)
fig.update_traces(marker_color= 'darkorange', row=1,col=2)
fig.update_traces(marker_color= 'slateblue', row=2,col=1)
fig.update_traces(marker_color= 'plum', row=2,col=2)
fig.update_layout(title='Correlation of RAM, Storage, Main, Second Camera & Actual Price', title_font_color="black", title_font_size =24, font_color="grey", font_size =12, height=400*2, width=1100)
fig.show()

# %%
#Visualize correlation matrix by heatmap (RAM (GB), Storage (GB), Main Camera, Second Camera and Actual Price) (optional)
plt.figure(figsize=(16,8))
sns.heatmap(df[selected_columns1].corr(), annot = True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of RAM (GB), Storage (GB), Main Camera, Second Camera & Actual Price', size =16, weight='semibold')
plt.show()

# %% [markdown]
# - RAM vs Price:
#    + There is a positive correlation between RAM size and the price of the mobile phone. As the RAM increases, the price tends to increase, especially in the higher RAM segments (10GB and above).
#    + Phones with around 8–12GB RAM show varying price points, indicating a wider price range for mid-to-high RAM phones.
# - Storage vs Price:
#    + Mobile phones with larger storage capacity tend to have higher prices. Phones with 128GB, 256GB, and 512GB storage are clustered around higher price points.
#    + Even within the same storage size (e.g., 128GB), there seems to be a large variation in price, possibly due to differences in other features like camera quality, brand, or performance.
# - Main Camera vs Price:
#    + Phones with higher main camera resolution (around 50 MP and 200 MP) generally fall in the higher price category.
#    + There’s a large cluster of phones around 12-64 MP for the main camera resolution, indicating this range is common for most phones, but price variations exist.
# - Second Camera vs Price:
#    + Similar to the main camera, phones with higher second camera resolution (10–50 MP) tend to be more expensive.
#    + Phones with dual cameras having higher resolutions on the second camera also seem to push the price upwards.

# %% [markdown]
# ### 5.4 Camera Analysis

# %%
#Handle null values in the Camera columns from the original (df_dup) one
df_cleaned_cam = df_dup.dropna(subset=['Main Camera'])
df_cleaned_cam['Second Camera'] = df_cleaned_cam['Second Camera'].replace(0,np.nan)
#Calculate percentage having one or two cameras
have_second_cam = df_cleaned_cam['Second Camera'].dropna()
percentage_two_cam = (len(have_second_cam)/len(df_cleaned_cam['Second Camera']))*100
percentage_two_cam = round(percentage_two_cam,2)
percentage_one_cam = 100 - percentage_two_cam
percentage_one_cam = round(percentage_one_cam,2)

value = [64.1,35.9]
name = ['One Camera', 'Two Cameras']
colors=['slateblue', 'plum']

fig = go.Figure(data=go.Pie(values=value, labels=name, marker_colors=colors))
fig.update_layout(title_font_size =24, title_text ='Percentage of Camera Trend', title_font_color="black", 
                  font_size=12,font_color="grey", height=500, template="plotly_white")

fig.show()
print(f'Percentage of Phones having two cameras: {percentage_two_cam}%')
print(f'Percentage of Phones having only one cameras: {percentage_one_cam}%')

# %% [markdown]
# The phones having only main camera is still more popular than those having two cameras. However, following the development of technology, it is likely that phones having two cammera will be common in the upcomming years.

# %%
#Most Common Camera Resolution
common_main_camera= df_dup['Main Camera'].value_counts().reset_index()
common_second_camera= df_dup['Second Camera'].value_counts().reset_index()
common_second_camera = common_second_camera[common_second_camera['Second Camera']!=0]
print(common_main_camera)
print(common_second_camera)

highest_main_cam = common_main_camera[common_main_camera['Main Camera']==50]
other_main_cam = common_main_camera[common_main_camera['Main Camera']!=50]

highest_second_cam = common_second_camera[common_second_camera['Second Camera']==2]
other_second_cam = common_second_camera[common_second_camera['Second Camera']!=2]

#Visualise by drawing 2 bar charts side by side to compare
fig = make_subplots(rows=1, cols=2, subplot_titles = ('Main Cam', 'Second Cam'))
fig.add_bar(x=highest_main_cam['Main Camera'], y=highest_main_cam['count'], textposition='outside',
            marker_color='slateblue', text=highest_main_cam['count'], name='Most common main',row =1, col=1)
fig.add_bar(x=other_main_cam['Main Camera'], y=other_main_cam['count'], marker_color='lavender', name='Main cam resolution',row =1, col=1)
fig.add_bar(x=highest_second_cam['Second Camera'], y=highest_second_cam['count'], textposition='outside',
            marker_color='plum', text=highest_second_cam['count'], name='Most common second',row =1, col=2)
fig.add_bar(x=other_second_cam['Second Camera'], y=other_second_cam['count'], marker_color='thistle', name='Second cam resolution', opacity=0.7,row =1, col=2)

fig.update_xaxes(title_text ='Main Cam Resolution', type='category', row=1, col=1)
fig.update_xaxes(title_text ='Second Cam Resolution', type='category', row=1, col=2)
fig.update_yaxes(title_text ='Count', row=1, col=1)
fig.update_yaxes(title_text ='Count', row=1, col=2)
fig.update_layout(template='plotly_white', title='Most Common Camera Resolution', font_color="grey", font_size =12, title_font_color="black", title_font_size =24, height=600)
fig.show()
print('The most common main camera resolution is 50MP, while he most common second camera resolution is 2MP')

# %% [markdown]
# - Main Cam Analysis:
#    + 50MP main cam is the most common among the devices, with a count of 480, indicating it's a common choice for many consumers
#    + This is followed by 8 and 3MP resolution, although the figures are much lower than 50 MP
#    + High resolution like 200MP and Low resolution like 48MP are far less common, possibly indicating that they are either higher-end options or less in demand
# 
# - Second Cam Analysis:
#    + 2MP is the most common, with a count of 219, suggesting it is the preferred option for many buyers
#    + This is followed by 8 and 12MP resolution, although the figures are not as significant as that of 2 MP
#    + Higher second cam resolution like 48 and 64MP are much less common, possibly due to higher price points or being niche products

# %%
#Calculate average star category of main and second cameras
main_star = df_dup.groupby('Main Camera')['Stars'].mean().round(1).sort_values().reset_index()
second_star = df_dup.groupby('Second Camera')['Stars'].mean().round(1).sort_values().reset_index()
second_star = second_star[second_star['Second Camera']!=0]
main_star['Star Category'] = pd.cut(main_star['Stars'], bins=[0, 3.4, 3.8, 4.2, 4.6, 5], labels=['Poor', 'Not Preferred', 'Fair', 'Good', 'Excellent'])
second_star['Star Category'] = pd.cut(second_star['Stars'], bins=[0, 3.4, 3.8, 4.2, 4.6, 5], labels=['Poor', 'Not Preferred', 'Fair', 'Good', 'Excellent'])
print(main_star)
print(second_star)

not_preferred_main = main_star[main_star['Star Category']=='Not Preferred']
fair_main = main_star[main_star['Star Category']=='Fair']
good_main = main_star[main_star['Star Category']=='Good']

fair_second = second_star[second_star['Star Category']=='Fair']
good_second = second_star[second_star['Star Category']=='Good']

#Visualize average star category of main and second cameras
#To see which cam resolution is most preferred
fig=make_subplots(rows=1,cols=2, subplot_titles=('Main Camera', 'Second Camera'))
fig.add_bar(x=not_preferred_main['Main Camera'], y=not_preferred_main['Stars'], marker_color='darkgrey', name='Not Preferred Main Cam',row=1,col=1)
fig.add_bar(x=fair_main['Main Camera'], y=fair_main['Stars'], marker_color='slateblue', opacity=0.5, name='Fair Main Cam',row=1,col=1)
fig.add_bar(x=good_main['Main Camera'], y=good_main['Stars'], marker_color='slateblue', name='Good Main Cam',row=1,col=1)
fig.add_bar(x=fair_second['Second Camera'], y=fair_second['Stars'], marker_color='plum',opacity=0.5, name='Fair Second Cam',row=1,col=2)
fig.add_bar(x=good_second['Second Camera'], y=good_second['Stars'], marker_color='plum', name='Good Second Cam',row=1,col=2)

fig.update_xaxes(title_text='Main camera resolution', type='category', row=1,col=1)
fig.update_yaxes(title_text='Stars', type='category', categoryorder='array', categoryarray=[0, 3.6, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6],row=1,col=1)
fig.update_yaxes(title_text='Stars', type='category', categoryorder='array', categoryarray=[0, 4.0, 4.2, 4.3, 4.4,4.5], row=1,col=2)
fig.update_xaxes(title_text='Second camera resolution', type='category',row=1,col=2)
fig.update_layout(title='Star Category of Camera Resolution',title_font_size=24, title_font_color='black', font_size=12, font_color='grey')
fig.show()


# %% [markdown]
# - Main Camera Resolution vs Stars Rating:
#    + Phones with main camera resolutions of 32MP and above (except for 12MP) have higher user ratings (above 4.3 stars).
#    + Lower camera resolutions (below 16 MP) tend to receive lower ratings, suggesting that camera quality heavily influences user satisfaction.
# - Second Camera Resolution vs Stars Rating:
#    + Similarly, higher second camera resolutions correspond with higher user ratings, with a peak around 4.5 stars.
#    + Lower-resolution second cameras (5, 16, and 20MP) receive comparatively just fair ratings.

# %%
#Calculate average actual price for main and second cam
#To see if the resolution of each cam affect the price or not
main_cam_price = df_dup.groupby('Main Camera')[['Actual price ₹','Discount price ₹']].mean().round(2).reset_index()
main_cam_price = main_cam_price.rename(columns={'Actual price ₹':'Main cam actual price', 'Discount price ₹': 'Main cam discount price'})
second_cam_price = df_dup.groupby('Second Camera')[['Actual price ₹','Discount price ₹']].mean().round(2).reset_index()
second_cam_price = second_cam_price.rename(columns={'Actual price ₹':'Second cam actual price', 'Discount price ₹': 'Second cam discount price'})
second_cam_price = second_cam_price[second_cam_price['Second Camera']!=0]
avg_main_second_cam_price = pd.concat([main_cam_price, second_cam_price],axis=1)
print(avg_main_second_cam_price)

#Visualize average actual price for main and second cam
colors_main_actual = ['silver',] * len(main_cam_price)
colors_main_actual[11] = 'slateblue'

colors_main_discount = ['gainsboro',] * len(main_cam_price)
colors_main_discount[11] = 'slateblue'

colors_second_actual = ['silver',] *  len(avg_main_second_cam_price)
colors_second_actual[10] = 'plum'

colors_second_discount = ['gainsboro',] *  len(avg_main_second_cam_price)
colors_second_discount[4] = 'plum'

fig = make_subplots(rows=1, cols=2, subplot_titles=('Main Camera','Second Camera'))
fig.add_bar(x=avg_main_second_cam_price['Main Camera'], y=avg_main_second_cam_price['Main cam actual price'], text = avg_main_second_cam_price['Main cam actual price'], marker_color=colors_main_actual, name='Actual Price', row=1, col=1)
fig.add_bar(x=avg_main_second_cam_price['Main Camera'], y=avg_main_second_cam_price['Main cam discount price'], text = avg_main_second_cam_price['Main cam discount price'], marker_color=colors_main_discount,opacity=0.7 ,name='Discount Price', row=1, col=1)
fig.add_bar(x=avg_main_second_cam_price['Second Camera'], y=avg_main_second_cam_price['Second cam actual price'], text = avg_main_second_cam_price['Second cam actual price'], marker_color=colors_second_actual, showlegend=False, row=1, col=2)
fig.add_bar(x=avg_main_second_cam_price['Second Camera'], y=avg_main_second_cam_price['Second cam discount price'], text = avg_main_second_cam_price['Second cam discount price'], marker_color=colors_second_discount, opacity=0.7,showlegend=False, row=1, col=2)

fig.update_traces(textposition='outside', texttemplate='%{text:.2f%}')
fig.update_xaxes(title_text ='Main cam resolution', type='category', row=1, col=1)
fig.update_xaxes(title_text ='Second cam resolution', type='category', row=1, col=2)
fig.update_yaxes(title_text ='Price', row=1, col=1)
fig.update_yaxes(title_text ='Price', row=1, col=2)
fig.update_layout(font_color="grey", template='plotly_white', font_size =10, title='Average Actual & Discount Price for Cam Resolution', title_font_color="black", title_font_size =24, height=600)
fig.show()

print('For the main camera resolution, the one having the highest mean actual and discount price is 200MP')
print('For the second camera resolution, the one having the highest mean actual price is 48MP, while that of discount price is 10MP')

# %% [markdown]
# - Main Camera Price Comparison:
#    + Phones with 200 MP cameras have the highest average actual and discount prices, indicating they are likely flagship devices with cutting-edge features.
#    + Phones with mid-range camera resolutions (50 MP, 64 MP) also have higher prices, but there are budget options available with 12 MP and 32 MP cameras.
# - Second Camera Price Comparison:
#    + For second cameras, phones with 48 MP have high actual prices, while phones with 10 MP second cameras see the highest discounts.
#    + Phones with mid-range second cameras (around 12 MP) tend to be more affordable.

# %%
#Calculate average discount % of main and second cam resolution
main_cam_discount_amount = df_dup.groupby('Main Camera')['Discount amount (%)'].mean().round(2).sort_values(ascending=False).reset_index()
second_cam_discount_amount = df_dup.groupby('Second Camera')['Discount amount (%)'].mean().round(2).sort_values(ascending=False).reset_index()
second_cam_discount_amount =second_cam_discount_amount[second_cam_discount_amount['Second Camera']!=0]
print(main_cam_discount_amount)
print(second_cam_discount_amount)

#Visualize average discount % of main and second cam resolution
fig = make_subplots(rows=1, cols=2, subplot_titles=('Main Camera','Second Camera'))
fig.add_bar(x=main_cam_discount_amount['Main Camera'], y=main_cam_discount_amount['Discount amount (%)'], name='Main Cam', text = main_cam_discount_amount['Discount amount (%)'], marker_color ='slateblue', row=1, col=1)
fig.add_bar(x=second_cam_discount_amount['Second Camera'], y=second_cam_discount_amount['Discount amount (%)'], name ='Second Cam', text=second_cam_discount_amount['Discount amount (%)'], marker_color='plum', row=1, col=2)

fig.update_traces(textposition='outside', texttemplate='%{text:.2f%}')
fig.update_xaxes(title_text ='Main cam resolution', type='category', row=1, col=1)
fig.update_xaxes(title_text ='Second cam resolution', type='category', row=1, col=2)
fig.update_yaxes(title_text ='Discount Amount', row=1, col=1)
fig.update_yaxes(title_text ='Discount Amount', row=1, col=1)
fig.update_layout(template='plotly_white', font_color="grey", font_size =10, title='Average Discount Amount of Cam Resolution', title_font_color="black", title_font_size =24, height=600)
fig.show()


# %% [markdown]
# - Main Camera generally has a higher discount amount compared to the Second Camera, regardless of resolution. The highest discount amount for the Main Camera is observed at 16MP resolution, with an average discount of 30, while the lowest is at 50MP resolution, with an average discount of 10.24.
# 
# - Second Camera shows a decreasing trend in discount amount as the resolution increases. The highest discount amount for the Second Camera is observed at 48MP resolution, with an average discount of 43.53, while the lowest is at 10 resolution, with an average discount of -54.75

# %% [markdown]
# ## 6. Feature Engineering

# %%
df= df.drop(columns=['Camera', 'Product Name'])
print(df.info())
#Encode categorical data
categorical_col = ['Brand', 'Price Category', 'Star Category']
le = LabelEncoder()
df['Brand'] = le.fit_transform(df['Brand'])
df['Price Category'] = le.fit_transform(df['Price Category'])
df['Star Category'] = le.fit_transform(df['Star Category'])

#Scale data before modelling
x = df.drop('Actual price ₹', axis =1)
y = df['Actual price ₹']
scaler_x= StandardScaler()
x = pd.DataFrame(scaler_x.fit_transform(x), columns= x.columns)
scaler_y= StandardScaler()
y = scaler_y.fit_transform(y.values.reshape(-1, 1))
print(f'Shape of x: {x.shape}')
print(f'Shape of y: {y.shape}')

# %% [markdown]
# ## 7. Model Selection and Evaluation

# %%
#Split data into the train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Linear Regression
lr_model = LinearRegression()
lr_model.fit(x_train, y_train)
y_pred_lr = lr_model.predict(x_test)
#Evaluate linear regression model using mse and r2
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print(f'MSE Linear Regression:{mse_lr}, and R² Score:{r2_lr}')


# Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(x_train, y_train)
y_pred_dt = dt_model.predict(x_test)
#Evaluate tree regressor using mse and r2
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)
print(f'MSE Decision Tree Regressor: {mse_dt}, and R² Score:{r2_dt}')

# Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(x_train, y_train)
y_pred_rf = rf_model.predict(x_test)
#Evaluate forest regressor using mse and r2
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f'MSE Random Forest Regressor: {mse_rf}, and R² Score:{r2_rf}')

# %% [markdown]
# Comment:
# - Lower MSE values indicate better model performance, so both the Decision Tree and Random Forest regressors are performing significantly better than the Linear Regression model
# - An R² score close to 1 indicates that the model explains a high proportion of the variance in the dependent variable. The Random Forest Regressor has an R² score of approximately 0.9913, which is excellent, suggesting that it explains over 99% of the variance
# 
# Overall, both the Decision Tree and Random Forest regressors demonstrate strong performance with low MSE values and high R² scores. The Random Forest Regressor, in particular, shows the best performance among the three. In short, the Random Forest model would be the best choice based on these metrics.
# 

# %% [markdown]
# 

# %% [markdown]
# ## 8. Business Insights and Recommendations

# %% [markdown]
# ### 8.1 Comprehensive Business Insights from Mobile Sales Data
# **1. Brand-Specific Strategies and Insights**
# - Premium Brands: Apple, Google, and Samsung:
#   + These brands maintain high price points with limited discounts, relying on a strategy focused on premium features, brand loyalty, and superior product quality.
#   + Their high star ratings (above 4.3) reflect strong customer satisfaction and a focus on the customer experience.
#   + This approach helps preserve their premium brand image and ensures profitability without engaging in aggressive price competition.
# - Recommendation:
#   + Continue focusing on innovation and introducing exclusive features that set them apart.
#   + Utilize limited-time discounts during major shopping events like Black Friday or festive sales to create urgency and boost sales without compromising their premium image.
#   + Offering extended warranties or premium service packages could further enhance customer loyalty and justify their pricing strategy, especially for high-end models.
# 
# 
# - Mid-Range Brands: Realme, Oppo, and Vivo:
#   + These brands focus on the mid-range market, balancing price and performance with moderate discount strategies.
#   + They maintain good star ratings, which suggests customer satisfaction is generally positive, though not as high as the premium segment.
#   + Their market position makes them vulnerable to competition from both premium brands, which offer better features at higher prices, and budget brands, which appeal to price-sensitive customers.
# - Recommendation:
#   + Enhance brand differentiation by highlighting unique features or user-friendly innovations that resonate with the target audience.
#   + Continue offering moderate discounts but avoid over-reliance on price cuts, as it could erode perceived value.
#   + Strengthen after-sales service and customer engagement programs to build loyalty and differentiate from both premium and budget competitors.
# 
# 
# - Budget Brands: Karbonn, Kechaoda, and Vox:
#   + These brands compete primarily on low price points, often accompanied by hefty discounts to attract price-sensitive customers.
#   + However, they tend to have lower star ratings, indicating potential quality issues or gaps in customer satisfaction.
#   + This can limit their sales potential and result in weaker brand loyalty, as customers may prioritize savings but become dissatisfied with product quality over time.
# 
# - Recommendation:
#   + Focus on improving product quality and addressing common customer complaints, as even small improvements could positively impact star ratings.
#   + Emphasize value-for-money features in marketing campaigns, such as battery life or display size, which are appealing at lower price points.
#   + Consider bundling devices with accessories or basic service packages to create a sense of added value, even at low price points.
# 
# **2. Market Segmentation by RAM and Storage Preferences**
# - Most Common RAM & Storage:
#   + 8 GB RAM is the most popular configuration, with a count of 361, indicating a strong preference for balanced performance.
#   + For storage, 128 GB is the leading choice, with 421 units, suggesting it hits the right balance between capacity and affordability for most consumers.
#   + Lesser-used options include higher RAM configurations (e.g., 32 GB and 64 GB) and higher storage capacities like 512 GB, which are typically reserved for more premium models.
# 
# - Customer Satisfaction Trends:
#   + Devices with 8 to 16 GB RAM tend to receive higher star ratings (between 4.2 and 4.4 stars), suggesting that these specifications are more in line with customer needs for multitasking and performance.
#   + For storage, ratings above 4 stars are associated with devices offering 64 GB and higher, indicating that sufficient storage is a key factor in customer satisfaction.
# 
# **3. Market Segmentation by Main and Second Cam Preferences**
# - Main Camera:
#    + 50MP main cameras are standard in mid-range devices, offering excellent image quality for the price. Higher-end models should focus on more advanced camera features.
#    + Recommendation: Mid-range brands should continue using 50MP cameras but emphasize software improvements (AI, night mode). Premium brands should market multi-camera setups and advanced features like optical zoom and image stabilization.
# - Second Camera:
#    + 2MP second cameras are common but not particularly valued, while premium devices offer better secondary camera options for versatile photography.
#    + Recommendation: For mid-range brands, upgrading to 8MP+ secondary cameras could enhance the photography experience, while premium brands should promote the multi-camera versatility for content creators.
# 
# **4. Overall Market Dynamics and Consumer Preferences**
# - Demand Concentration: The highest demand is in the mid-range market with 8 GB RAM and 128 GB storage devices, balancing affordability with adequate performance for most users.
# - Star Ratings as a Critical Factor: Higher star ratings correlate with configurations that balance performance and capacity. This indicates that customer satisfaction is closely tied to both hardware specifications and overall user experience.
# - Segmentation Based on Price Sensitivity: Premium brands rely on brand prestige and innovative features, while mid-range brands focus on affordability with a fair balance of quality. Budget brands compete primarily on price but struggle with perceived quality.

# %% [markdown]
# ### 8.2 Strategic Recommendations
# **1. For Premium Brands:**
# - Focus on Exclusive Offerings: Continue emphasizing features like advanced cameras, proprietary software enhancements, and high-end design.
# - Limited-Time Promotions: Implement strategic discounts during peak shopping seasons to spur demand without diluting the premium brand image.
# - Enhance Customer Support: Providing options like AppleCare, Samsung Premium Care, or Google Preferred Care can help maintain loyalty and justify higher prices.
# 
# **2. For Mid-Range Brands:**
# - Differentiate on Value: Highlight unique features like high-refresh-rate displays or fast-charging capabilities that set them apart from both budget and premium competitors.
# - Loyalty Programs: Implement customer loyalty programs that encourage repeat purchases and positive reviews, boosting word-of-mouth.
# - Balance Discounts with Quality: Continue offering discounts but ensure product quality remains high to prevent a decline in ratings.
# 
# **3. For Budget Brands:**
# - Quality Focused Improvements: Invest in quality control to address issues that lead to lower ratings, such as battery life or build quality.
# - Emphasize Affordability in Marketing: Highlight affordability and essential features that meet basic needs to appeal to cost-conscious buyers.
# - Offer Bundled Value: Providing affordable accessories or simple warranties can create a perception of added value, making budget devices more attractive.
# 
# **Conclusion:**
# The mobile market is characterized by distinct segments, each with unique strategies and consumer needs. Premium brands should focus on maintaining their high-quality image while using strategic promotions. Mid-range brands need to balance competitive pricing with quality offerings to maintain their position, while budget brands should aim for improved quality and value bundling to overcome challenges with customer satisfaction.
# 
# By tailoring strategies to these insights, brands can better align their offerings with customer expectations, improving sales performance and maintaining market relevance.


