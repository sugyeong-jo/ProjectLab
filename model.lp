\Problem name: Unist_optimization

Minimize
OBJROW: 2 u_a,0 + u_a,1 + u_a,2 + u_a,3 + u_a,4 + 2 u_a,5 + u_b,0 + u_b,1 + u_b,2 + u_b,3
 + u_b,4 + u_b,5 + 2 u_c,0 + u_c,1 + 2 u_c,2 + 2 u_c,3 + u_c,4 + u_c,5 + 2 u_d,0 + 2 u_d,1
 + 2 u_d,2 + 2 u_d,3 + 2 u_d,4 + u_d,5
Subject To
constr(0):  x_a,A,0 + x_a,B,0 + x_a,C,0 + u_a,0 = 3
constr(1):  x_a,A,1 + x_a,B,1 + x_a,C,1 + u_a,1 = 3
constr(2):  x_a,A,2 + x_a,B,2 + x_a,C,2 + u_a,2 = 4
constr(3):  x_a,A,3 + x_a,B,3 + x_a,C,3 + u_a,3 = 6
constr(4):  x_a,A,4 + x_a,B,4 + x_a,C,4 + u_a,4 = 2
constr(5):  x_a,A,5 + x_a,B,5 + x_a,C,5 + u_a,5 = 5
constr(6):  x_b,A,0 + x_b,B,0 + x_b,C,0 + u_b,0 = 4
constr(7):  x_b,A,1 + x_b,B,1 + x_b,C,1 + u_b,1 = 3
constr(8):  x_b,A,2 + x_b,B,2 + x_b,C,2 + u_b,2 = 2
constr(9):  x_b,A,3 + x_b,B,3 + x_b,C,3 + u_b,3 = 8
constr(10):  x_b,A,4 + x_b,B,4 + x_b,C,4 + u_b,4 = 3
constr(11):  x_b,A,5 + x_b,B,5 + x_b,C,5 + u_b,5 = 6
constr(12):  x_c,A,0 + x_c,B,0 + x_c,C,0 + u_c,0 = 5
constr(13):  x_c,A,1 + x_c,B,1 + x_c,C,1 + u_c,1 = 7
constr(14):  x_c,A,2 + x_c,B,2 + x_c,C,2 + u_c,2 = 9
constr(15):  x_c,A,3 + x_c,B,3 + x_c,C,3 + u_c,3 = 1
constr(16):  x_c,A,4 + x_c,B,4 + x_c,C,4 + u_c,4 = 5
constr(17):  x_c,A,5 + x_c,B,5 + x_c,C,5 + u_c,5 = 9
constr(18):  x_d,A,0 + x_d,B,0 + x_d,C,0 + u_d,0 = 6
constr(19):  x_d,A,1 + x_d,B,1 + x_d,C,1 + u_d,1 = 8
constr(20):  x_d,A,2 + x_d,B,2 + x_d,C,2 + u_d,2 = 1
constr(21):  x_d,A,3 + x_d,B,3 + x_d,C,3 + u_d,3 = 5
constr(22):  x_d,A,4 + x_d,B,4 + x_d,C,4 + u_d,4 = 8
constr(23):  x_d,A,5 + x_d,B,5 + x_d,C,5 + u_d,5 = 2
constr(24):  x_a,A,0 + x_b,A,0 + x_c,A,0 + x_d,A,0 <= 1200
constr(25):  x_a,A,1 + x_b,A,1 + x_c,A,1 + x_d,A,1 <= 1200
constr(26):  x_a,A,2 + x_b,A,2 + x_c,A,2 + x_d,A,2 <= 1200
constr(27):  x_a,A,3 + x_b,A,3 + x_c,A,3 + x_d,A,3 <= 1200
constr(28):  x_a,A,4 + x_b,A,4 + x_c,A,4 + x_d,A,4 <= 1200
constr(29):  x_a,A,5 + x_b,A,5 + x_c,A,5 + x_d,A,5 <= 1200
constr(30):  x_a,B,0 + x_b,B,0 + x_c,B,0 + x_d,B,0 <= 1200
constr(31):  x_a,B,1 + x_b,B,1 + x_c,B,1 + x_d,B,1 <= 1200
constr(32):  x_a,B,2 + x_b,B,2 + x_c,B,2 + x_d,B,2 <= 1200
constr(33):  x_a,B,3 + x_b,B,3 + x_c,B,3 + x_d,B,3 <= 1200
constr(34):  x_a,B,4 + x_b,B,4 + x_c,B,4 + x_d,B,4 <= 1200
constr(35):  x_a,B,5 + x_b,B,5 + x_c,B,5 + x_d,B,5 <= 1200
constr(36):  x_a,C,0 + x_b,C,0 + x_c,C,0 + x_d,C,0 <= 1200
constr(37):  x_a,C,1 + x_b,C,1 + x_c,C,1 + x_d,C,1 <= 1200
constr(38):  x_a,C,2 + x_b,C,2 + x_c,C,2 + x_d,C,2 <= 1200
constr(39):  x_a,C,3 + x_b,C,3 + x_c,C,3 + x_d,C,3 <= 1200
constr(40):  x_a,C,4 + x_b,C,4 + x_c,C,4 + x_d,C,4 <= 1200
constr(41):  x_a,C,5 + x_b,C,5 + x_c,C,5 + x_d,C,5 <= 1200
constr(42):  x_a,A,0 >= -0
constr(43):  x_a,A,1 >= -0
constr(44):  x_a,A,2 >= -0
constr(45):  x_a,A,3 >= -0
constr(46):  x_a,A,4 >= -0
constr(47):  x_a,A,5 >= -0
constr(48):  x_a,B,0 >= -0
constr(49):  x_a,B,1 >= -0
constr(50):  x_a,B,2 >= -0
constr(51):  x_a,B,3 >= -0
constr(52):  x_a,B,4 >= -0
constr(53):  x_a,B,5 >= -0
constr(54):  x_a,C,0 >= -0
constr(55):  x_a,C,1 >= -0
constr(56):  x_a,C,2 >= -0
constr(57):  x_a,C,3 >= -0
constr(58):  x_a,C,4 >= -0
constr(59):  x_a,C,5 >= -0
constr(60):  x_b,A,0 >= -0
constr(61):  x_b,A,1 >= -0
constr(62):  x_b,A,2 >= -0
constr(63):  x_b,A,3 >= -0
constr(64):  x_b,A,4 >= -0
constr(65):  x_b,A,5 >= -0
constr(66):  x_b,B,0 >= -0
constr(67):  x_b,B,1 >= -0
constr(68):  x_b,B,2 >= -0
constr(69):  x_b,B,3 >= -0
constr(70):  x_b,B,4 >= -0
constr(71):  x_b,B,5 >= -0
constr(72):  x_b,C,0 >= -0
constr(73):  x_b,C,1 >= -0
constr(74):  x_b,C,2 >= -0
constr(75):  x_b,C,3 >= -0
constr(76):  x_b,C,4 >= -0
constr(77):  x_b,C,5 >= -0
constr(78):  x_c,A,0 >= -0
constr(79):  x_c,A,1 >= -0
constr(80):  x_c,A,2 >= -0
constr(81):  x_c,A,3 >= -0
constr(82):  x_c,A,4 >= -0
constr(83):  x_c,A,5 >= -0
constr(84):  x_c,B,0 >= -0
constr(85):  x_c,B,1 >= -0
constr(86):  x_c,B,2 >= -0
constr(87):  x_c,B,3 >= -0
constr(88):  x_c,B,4 >= -0
constr(89):  x_c,B,5 >= -0
constr(90):  x_c,C,0 >= -0
constr(91):  x_c,C,1 >= -0
constr(92):  x_c,C,2 >= -0
constr(93):  x_c,C,3 >= -0
constr(94):  x_c,C,4 >= -0
constr(95):  x_c,C,5 >= -0
constr(96):  x_d,A,0 >= -0
constr(97):  x_d,A,1 >= -0
constr(98):  x_d,A,2 >= -0
constr(99):  x_d,A,3 >= -0
constr(100):  x_d,A,4 >= -0
constr(101):  x_d,A,5 >= -0
constr(102):  x_d,B,0 >= -0
constr(103):  x_d,B,1 >= -0
constr(104):  x_d,B,2 >= -0
constr(105):  x_d,B,3 >= -0
constr(106):  x_d,B,4 >= -0
constr(107):  x_d,B,5 >= -0
constr(108):  x_d,C,0 >= -0
constr(109):  x_d,C,1 >= -0
constr(110):  x_d,C,2 >= -0
constr(111):  x_d,C,3 >= -0
constr(112):  x_d,C,4 >= -0
constr(113):  x_d,C,5 >= -0
constr(114):  u_a,0 >= -0
constr(115):  u_a,1 >= -0
constr(116):  u_a,2 >= -0
constr(117):  u_a,3 >= -0
constr(118):  u_a,4 >= -0
constr(119):  u_a,5 >= -0
constr(120):  u_b,0 >= -0
constr(121):  u_b,1 >= -0
constr(122):  u_b,2 >= -0
constr(123):  u_b,3 >= -0
constr(124):  u_b,4 >= -0
constr(125):  u_b,5 >= -0
constr(126):  u_c,0 >= -0
constr(127):  u_c,1 >= -0
constr(128):  u_c,2 >= -0
constr(129):  u_c,3 >= -0
constr(130):  u_c,4 >= -0
constr(131):  u_c,5 >= -0
constr(132):  u_d,0 >= -0
constr(133):  u_d,1 >= -0
constr(134):  u_d,2 >= -0
constr(135):  u_d,3 >= -0
constr(136):  u_d,4 >= -0
constr(137):  u_d,5 >= -0
Bounds
Integers
x_a,A,0 x_a,A,1 x_a,A,2 x_a,A,3 x_a,A,4 x_a,A,5 x_a,B,0 x_a,B,1 x_a,B,2 x_a,B,3 
x_a,B,4 x_a,B,5 x_a,C,0 x_a,C,1 x_a,C,2 x_a,C,3 x_a,C,4 x_a,C,5 x_b,A,0 x_b,A,1 
x_b,A,2 x_b,A,3 x_b,A,4 x_b,A,5 x_b,B,0 x_b,B,1 x_b,B,2 x_b,B,3 x_b,B,4 x_b,B,5 
x_b,C,0 x_b,C,1 x_b,C,2 x_b,C,3 x_b,C,4 x_b,C,5 x_c,A,0 x_c,A,1 x_c,A,2 x_c,A,3 
x_c,A,4 x_c,A,5 x_c,B,0 x_c,B,1 x_c,B,2 x_c,B,3 x_c,B,4 x_c,B,5 x_c,C,0 x_c,C,1 
x_c,C,2 x_c,C,3 x_c,C,4 x_c,C,5 x_d,A,0 x_d,A,1 x_d,A,2 x_d,A,3 x_d,A,4 x_d,A,5 
x_d,B,0 x_d,B,1 x_d,B,2 x_d,B,3 x_d,B,4 x_d,B,5 x_d,C,0 x_d,C,1 x_d,C,2 x_d,C,3 
x_d,C,4 x_d,C,5 u_a,0 u_a,1 u_a,2 u_a,3 u_a,4 u_a,5 u_b,0 u_b,1 
u_b,2 u_b,3 u_b,4 u_b,5 u_c,0 u_c,1 u_c,2 u_c,3 u_c,4 u_c,5 
u_d,0 u_d,1 u_d,2 u_d,3 u_d,4 u_d,5 
End
