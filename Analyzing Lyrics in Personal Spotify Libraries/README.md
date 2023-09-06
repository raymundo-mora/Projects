# Spotify Topic Models 


## 1.0 Getting Started 
If you want to analyze your own *Spotify* data using the same methods you can do so by following the steps laid out ahead.

### 1.0.1 Spotify API
To be able to access your own songs using the *Spoitfy* API. You can follow the steps from the official documentation [here](https://developer.spotify.com/documentation/web-api). Once you have suffessfully created a WebApp, you should be able to look at your credentials in the settings page of your projecct similar to the following: 

<br>

<img width="600" alt="image" src="https://user-images.githubusercontent.com/92943544/230748854-e50372d0-cd5c-4aa5-bd49-63466c95c4e2.png">



### 1.0.2 Genius API
Similarly, you will need to create a project with a *Genius* account. To get started you can follow the instructions in the official documentation [here](https://docs.genius.com/#/getting-started-h1). Once your project has been created you can see your projects at https://genius.com/api-clients . Your credentials should be displayed in a page similar to the following.

<img width="647" alt="image" src="https://user-images.githubusercontent.com/92943544/230749121-30a44e69-411f-4da6-b657-488f2bcb6f31.png">


### 1.0.3

Once you have generated credentials for both of the applications you can create a file named `.env` in the root folder of this project to create your environment variables. I found this way of storing my credentials to be the easiest way for managing my project and protecting my credentials. Make sure to include `.env` in your `.gitignore` file before publishing your project anywhere. You will need to name your environment variables exactly as below. Once this step has been completed you are ready to create your own dataset by running the code in `DataCollection.ipynb`


<img width="447" alt="image" src="https://user-images.githubusercontent.com/92943544/230749012-8df6c051-a5b0-47e4-931c-bc484b405336.png">
