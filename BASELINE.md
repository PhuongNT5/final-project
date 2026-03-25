Table 4 AP score and TP and FP values obtained for each class in the test process with the YOLO v4 network
From: Unsafe-Net: YOLO v4 and ConvLSTM based computer vision system for real-time detection of unsafe behaviours in workplace

Unsafe/safe behaviour

AP (%)

TP

FP

Safe Walkway Violation

97.89

97

12

Unauthorized Intervention

95.83

23

0

Opened Panel Cover

96.92

63

7

Carrying Overload with Forklift

97.50

40

2

Safe Walkway

90.82

36

5

Authorized Intervention

100

12

0

Closed Panel Cover

100

41

0

Safe Carrying

86.30

11

8


Table 5 Performance of ConvLSTM network without video frame reduction with YOLO v4 for each unsafe behaviour class for the video test set
From: Unsafe-Net: YOLO v4 and ConvLSTM based computer vision system for real-time detection of unsafe behaviours in workplace

 	
Prediction

 	
Class ID

0

1

2

3

Actual

0

31

0

0

1

1

0

9

2

0

2

0

3

10

0

3

1

0

0

7


Table 6 Performance of the Unsafe-Net (YOLO v4 + ConvLSTM) network proposed in this study for each unsafe behaviour class for the video test set
From: Unsafe-Net: YOLO v4 and ConvLSTM based computer vision system for real-time detection of unsafe behaviours in workplace

 	
Prediction

 	
Class ID

0

1

2

3

Actual

0

32

0

0

0

1

0

10

1

0

2

0

1

12

0

3

0

0

0

8

Table 7 Comparison of the performance of the ConvLSTM network and the proposed Unsafe-Net model for the Acc metric in each unsafe behaviour class for the video test set
From: Unsafe-Net: YOLO v4 and ConvLSTM based computer vision system for real-time detection of unsafe behaviours in workplace

Class ID

Unsafe behaviour

Acc (%) via ConvLSTM

Acc (%) via Unsafe-Net (YOLO v4 + ConvLSTM)

0

Safe Walkway Violation

96.88

100

1

Unauthorized Intervention

81.82

90.91

2

Opened Panel Cover

76.92

92.31

3

Carrying Overload with Forklift

87.50

100

Overall (%)

85.78

95.81