# Machine Learning for COVID-19 Spread Prediction

## Introduction
Machine learning (ML) refers to a range of non-traditional programming methods through which information can be obtained directly from data manipulation. ML algorithms exploit various architectures. In this project, we utilize a multi-layer interconnected network structure, each populated by artificial neurons capable of processing input information to produce an output to transfer to the homologous computing units of the next layer. This architecture, inspired by the functioning of the human brain, defines the scope of Deep Neural Learning (neural networks). We employ a structured cell-based architecture known as Long Short-Term Memory (LSTM), a type of neural network characterized by the ability to selectively store or erase incoming information. In this project, we apply LSTM neural networks to predict the spread of Covid-19 at the Italian level.

## Thesis Overview
The first part of the project focuses on describing neural networks and the LSTM architecture. We introduce the concept of Loss function, which measures the algorithm's effectiveness, and Gradient Descent, which allows the machine to learn. We describe the characteristic structure of an LSTM network and its main components.

## Data Organization
In the second chapter, we describe how epidemiological and mobility data are organized, which we use to predict the spread of Covid-19. In particular, we make predictions on the number of patients in intensive care units with a prediction horizon covering 7 days. We introduce an index representing the unvaccinated population and evaluate its effect on the prediction. We then make predictions by modifying the vaccination index in the preceding days, assuming either a halt to the vaccination campaign or an increase in daily doses administered.

## Conclusion
In conclusion, we observe good accuracy in the predictions made by the network, which improves with the introduction of vaccination data. As expected, by varying the speed of the vaccination campaign, we observe that the curve describing the number of patients in intensive care units decreases more as the number of vaccine doses administered daily increases.
