import numpy as np

# data taken from McGPU files (Badal A, Badano A. Accelerating Monte Carlo simulations of photon transport in a voxelized geometry using a massively parallel graphics processing unit. Med Phys. 2009 Nov;36(11):4878–80. )

#[RAYLEIGH INTERACTIONS (RITA sampling  of atomic form factor from EPDL database)]
#[SAMPLING DATA FROM COMMON/CGRA/: X, P, A, B, ITL, ITU] <- ITL is lower bound for "hinted" binary search, ITU is upper bound for "hinted" binary search

water_RITA_PARAMS = np.array([
	[  0.0000000000E+00,  0.0000000000E+00, -4.0205185305E-02, -3.6052824066E-04,   1,   2],
	[  4.0825810621E-03,  2.3467282326E-02, -3.9554638952E-02, -1.8346193045E-04,   1,   2],
	[  8.1651621242E-03,  4.5080089702E-02, -7.4195584868E-02, -4.9346205269E-04,   1,   3],
	[  1.6330324248E-02,  8.3480769246E-02, -7.0260016507E-02, -3.2933139118E-04,   2,   3],
	[  2.4495486373E-02,  1.1648882951E-01, -6.6454925290E-02, -2.1962940078E-04,   2,   3],
	[  3.2660648497E-02,  1.4511200097E-01, -6.2860038358E-02, -1.3511098440E-04,   2,   4],
	[  4.0825810621E-02,  1.7013834002E-01, -5.9483282267E-02, -6.6925579352E-05,   3,   4],
	[  4.8990972745E-02,  1.9218872133E-01, -5.6315805245E-02, -1.1012651333E-05,   3,   4],
	[  5.7156134869E-02,  2.1175662346E-01, -5.3344507630E-02,  3.5042661140E-05,   3,   4],
	[  6.5321296993E-02,  2.2923773859E-01, -9.6336085821E-02,  3.5492233082E-04,   3,   4],
	[  8.1651621242E-02,  2.5916037549E-01, -8.7090411097E-02,  5.5745332905E-04,   3,   5],
	[  9.7981945490E-02,  2.8387900529E-01, -7.8828931474E-02,  6.8567651536E-04,   4,   5],
	[  1.1431226974E-01,  3.0470582128E-01, -7.1439646769E-02,  7.5976213917E-04,   4,   5],
	[  1.3064259399E-01,  3.2256039735E-01, -6.4825711282E-02,  7.9461245918E-04,   4,   5],
	[  1.4697291824E-01,  3.3810237877E-01, -5.8902704057E-02,  8.0137859259E-04,   4,   6],
	[  1.6330324248E-01,  3.5181489717E-01, -5.3596364054E-02,  7.8843488467E-04,   5,   6],
	[  1.7963356673E-01,  3.6405845499E-01, -4.8840891207E-02,  7.6206090657E-04,   5,   6],
	[  1.9596389098E-01,  3.7510659638E-01, -8.5636264803E-02,  2.8253439378E-03,   5,   6],
	[  2.2862453948E-01,  3.9441339045E-01, -7.2154979104E-02,  2.4836808841E-03,   5,   7],
	[  2.6128518797E-01,  4.1093431041E-01, -6.1164521239E-02,  2.1361261466E-03,   6,   7],
	[  2.9394583647E-01,  4.2543294670E-01, -5.2205507882E-02,  1.8124600678E-03,   6,   7],
	[  3.2660648497E-01,  4.3841598264E-01, -8.6250872434E-02,  5.5867367349E-03,   6,   8],
	[  3.9192778196E-01,  4.6112572456E-01, -6.5901107251E-02,  3.9007195078E-03,   7,   8],
	[  4.5724907895E-01,  4.8081919032E-01, -5.1950172456E-02,  2.7156852743E-03,   7,   8],
	[  5.2257037595E-01,  4.9845094785E-01, -4.2284433843E-02,  1.8965013144E-03,   7,   9],
	[  5.8789167294E-01,  5.1458137013E-01, -3.5497972330E-02,  1.3313569390E-03,   8,   9],
	[  6.5321296993E-01,  5.2955986148E-01, -3.0663849106E-02,  9.3947906014E-04,   8,  10],
	[  7.1853426693E-01,  5.4361528692E-01, -2.7169410283E-02,  6.6542736858E-04,   9,  10],
	[  7.8385556392E-01,  5.5690395643E-01, -2.4606296586E-02,  4.7186087174E-04,   9,  10],
	[  8.4917686091E-01,  5.6953661505E-01, -2.2699190071E-02,  3.3372714907E-04,   9,  11],
	[  9.1449815791E-01,  5.8159436474E-01, -4.1577440160E-02,  7.8700403536E-04,  10,  11],
	[  1.0451407519E+00,  6.0421637816E-01, -3.7840228866E-02,  3.5306941919E-04,  10,  11],
	[  1.1757833459E+00,  6.2511863663E-01, -3.5522085919E-02,  1.1458012236E-04,  10,  12],
	[  1.3064259399E+00,  6.4453183031E-01, -3.3999483495E-02, -1.9407619122E-05,  11,  12],
	[  1.4370685339E+00,  6.6262269241E-01, -3.2937141878E-02, -9.5531062104E-05,  11,  12],
	[  1.5677111278E+00,  6.7952053529E-01, -6.2204197259E-02, -6.0524832678E-04,  11,  12],
	[  1.8289963158E+00,  7.1014436348E-01, -6.0076024160E-02, -7.1107236183E-04,  11,  13],
	[  2.0902815038E+00,  7.3708380924E-01, -5.8453585550E-02, -7.2283597725E-04,  12,  13],
	[  2.3515666918E+00,  7.6087149512E-01, -5.7043467470E-02, -6.9999477301E-04,  12,  14],
	[  2.6128518797E+00,  7.8194397933E-01, -5.5732388096E-02, -6.6526468978E-04,  13,  14],
	[  2.8741370677E+00,  8.0066674542E-01, -5.4476302331E-02, -6.2768835104E-04,  13,  15],
	[  3.1354222557E+00,  8.1734897963E-01, -5.3258791667E-02, -5.9088469001E-04,  14,  15],
	[  3.3967074437E+00,  8.3225376941E-01, -5.2074528048E-02, -5.5622401717E-04,  14,  16],
	[  3.6579926316E+00,  8.4560584365E-01, -5.0922458651E-02, -5.2412214870E-04,  15,  16],
	[  3.9192778196E+00,  8.5759775246E-01, -4.9802959243E-02, -4.9459110166E-04,  15,  17],
	[  4.1805630076E+00,  8.6839492660E-01, -9.2692445871E-02, -1.8207005178E-03,  16,  17],
	[  4.7031333835E+00,  8.8695565707E-01, -8.8947025397E-02, -1.6370117965E-03,  16,  18],
	[  5.2257037595E+00,  9.0221197692E-01, -8.5427580944E-02, -1.4807028756E-03,  17,  19],
	[  5.7482741354E+00,  9.1485981792E-01, -8.2129703932E-02, -1.3464955076E-03,  18,  19],
	[  6.2708445114E+00,  9.2542877080E-01, -7.9043580275E-02, -1.2302723543E-03,  18,  19],
	[  6.7934148873E+00,  9.3432589924E-01, -7.6156566717E-02, -1.1288428234E-03,  18,  20],
	[  7.3159852633E+00,  9.4186712269E-01, -7.3454896837E-02, -1.0397147239E-03,  19,  20],
	[  7.8385556392E+00,  9.4829989494E-01, -7.0924707114E-02, -9.6091775325E-04,  19,  21],
	[  8.3611260152E+00,  9.5381974548E-01, -6.8552614471E-02, -8.9087451717E-04,  20,  21],
	[  8.8836963911E+00,  9.5858245889E-01, -6.6326015777E-02, -8.2830718074E-04,  20,  22],
	[  9.4062667670E+00,  9.6271312675E-01, -1.2025333078E-01, -2.9891714558E-03,  21,  23],
	[  1.0451407519E+01,  9.6946431133E-01, -1.1355069907E-01, -2.6217494010E-03,  22,  23],
	[  1.1496548271E+01,  9.7468022941E-01, -1.0753080631E-01, -2.3184688393E-03,  22,  23],
	[  1.2541689023E+01,  9.7877306903E-01, -1.0210013261E-01, -2.0651203600E-03,  22,  24],
	[  1.3586829775E+01,  9.8202926434E-01, -9.7179827230E-02, -1.8512588820E-03,  23,  24],
	[  1.4631970527E+01,  9.8465198605E-01, -9.2703536397E-02, -1.6690533971E-03,  23,  24],
	[  1.5677111278E+01,  9.8678802088E-01, -8.8615340055E-02, -1.5125324325E-03,  23,  25],
	[  1.6722252030E+01,  9.8854519174E-01, -1.5541454933E-01, -5.2774149980E-03,  24,  25],
	[  1.8812533534E+01,  9.9122491127E-01, -1.4430994080E-01, -4.4436837873E-03,  24,  26],
	[  2.0902815038E+01,  9.9312940731E-01, -1.3466167548E-01, -3.7932607328E-03,  25,  26],
	[  2.2993096542E+01,  9.9451996098E-01, -1.2620669310E-01, -3.2760314933E-03,  25,  27],
	[  2.5083378045E+01,  9.9555892109E-01, -2.0942778844E-01, -1.0767516522E-02,  26,  27],
	[  2.9263941053E+01,  9.9696487690E-01, -1.8990771678E-01, -8.4578231589E-03,  26,  28],
	[  3.3444504061E+01,  9.9783440299E-01, -1.7363663826E-01, -6.8199894983E-03,  27,  28],
	[  3.7625067068E+01,  9.9840086814E-01, -1.5988782775E-01, -5.6163136955E-03,  27,  29],
	[  4.1805630076E+01,  9.9878565258E-01, -2.5248935631E-01, -1.7467933697E-02,  28,  30],
	[  5.0166756091E+01,  9.9925186908E-01, -2.2500493520E-01, -1.2902765154E-02,  29,  30],
	[  5.8527882106E+01,  9.9950680014E-01, -2.0270868686E-01, -9.9222184421E-03,  29,  31],
	[  6.6889008121E+01,  9.9965782238E-01, -3.0045522919E-01, -2.8673669306E-02,  30,  32],
	[  8.3611260152E+01,  9.9981580178E-01, -2.6324846727E-01, -1.9588555959E-02,  31,  32],
	[  1.0033351218E+02,  9.9988969078E-01, -3.5603276491E-01, -5.0564469842E-02,  31,  32],
	[  1.3377801624E+02,  9.9995137189E-01, -4.1323737766E-01, -1.0533310764E-01,  31,  33],
	[  2.0066702436E+02,  9.9998491798E-01, -3.6057940862E-01, -5.3149874364E-02,  32,  33],
	[  2.6755603248E+02,  9.9999348400E-01, -3.5645472721E-01, -3.4628029631E-01,  32,  33],
	[  5.3511206497E+02,  9.9999915411E-01, -3.5248398363E-01, -3.5281611192E-01,  32,  34],
	[  1.0702241299E+03,  9.9999989203E-01, -3.5027375670E-01, -3.5637874339E-01,  33,  34],
	[  2.1404482599E+03,  9.9999998634E-01, -3.4903502325E-01, -3.5834192852E-01,  33,  35],
	[  4.2808965198E+03,  9.9999999828E-01, -3.4832817729E-01, -3.5944490982E-01,  34,  35],
	[  8.5617930395E+03,  9.9999999978E-01, -3.4791484443E-01, -3.6008026468E-01,  34,  35],
	[  1.7123586079E+04,  9.9999999997E-01, -3.4766636644E-01, -3.6045669628E-01,  34,  36],
	[  3.4247172158E+04,  1.0000000000E+00, -3.4751272205E-01, -3.6068630238E-01,  35,  36],
	[  6.8494344316E+04,  1.0000000000E+00, -3.4741515475E-01, -3.6083032758E-01,  35,  37],
	[  1.3698868863E+05,  1.0000000000E+00, -3.4735171673E-01, -3.6092298980E-01,  36,  37],
	[  2.7397737726E+05,  1.0000000000E+00, -3.4730964012E-01, -3.6098391820E-01,  36,  37],
	[  5.4795475453E+05,  1.0000000000E+00, -3.4728127937E-01, -3.6102470291E-01,  36,  37],
	[  1.0959095091E+06,  1.0000000000E+00, -3.4726192175E-01, -3.6105239255E-01,  36,  38],
	[  2.1918190181E+06,  1.0000000000E+00, -3.4724858223E-01, -3.6107139717E-01,  37,  38],
	[  4.3836380362E+06,  1.0000000000E+00, -3.4723932402E-01, -3.6108454797E-01,  37,  38],
	[  8.7672760725E+06,  1.0000000000E+00, -3.4723286462E-01, -3.6109370323E-01,  37,  39],
	[  1.7534552145E+07,  1.0000000000E+00, -3.4722834071E-01, -3.6110010508E-01,  38,  39],
	[  3.5069104290E+07,  1.0000000000E+00, -3.4722516362E-01, -3.6110459593E-01,  38,  39],
	[  7.0138208580E+07,  1.0000000000E+00, -3.4722292797E-01, -3.6110775347E-01,  38,  40],
	[  1.4027641716E+08,  1.0000000000E+00, -3.4722135258E-01, -3.6110997719E-01,  39,  40],
	[  2.8055283432E+08,  1.0000000000E+00, -4.1666626372E-01, -1.1419661441E-01,  39,  40],
	[  4.2082925148E+08,  1.0000000000E+00, -3.6554726827E-01, -5.6326737258E-02,  39,  41],
	[  5.6110566864E+08,  1.0000000000E+00, -3.1572167306E-01, -3.3610880438E-02,  40,  41],
	[  7.0138208580E+08,  1.0000000000E+00, -2.7549333792E-01, -2.2345537870E-02,  40,  42],
	[  8.4165850296E+08,  1.0000000000E+00, -2.4353878594E-01, -1.5935910489E-02,  41,  42],
	[  9.8193492012E+08,  1.0000000000E+00, -2.1787713995E-01, -1.1940127058E-02,  41,  43],
	[  1.1222113373E+09,  1.0000000000E+00, -1.9693754061E-01, -9.2806445461E-03,  42,  43],
	[  1.2624877544E+09,  1.0000000000E+00, -1.7957858165E-01, -7.4210882577E-03,  42,  44],
	[  1.4027641716E+09,  1.0000000000E+00, -1.6497928535E-01, -6.0697602706E-03,  43,  44],
	[  1.5430405888E+09,  1.0000000000E+00, -1.5254319090E-01, -5.0568591947E-03,  43,  45],
	[  1.6833170059E+09,  1.0000000000E+00, -1.4183004618E-01, -4.2780514873E-03,  44,  46],
	[  1.8235934231E+09,  1.0000000000E+00, -1.3250934668E-01, -3.6663427886E-03,  45,  46],
	[  1.9638698402E+09,  1.0000000000E+00, -1.2432887249E-01, -3.1771099158E-03,  45,  47],
	[  2.1041462574E+09,  1.0000000000E+00, -1.1709317681E-01, -2.7796959609E-03,  46,  47],
	[  2.2444226746E+09,  1.0000000000E+00, -1.1064866295E-01, -2.4524729506E-03,  46,  48],
	[  2.3846990917E+09,  1.0000000000E+00, -1.0487306517E-01, -2.1798278614E-03,  47,  48],
	[  2.5249755089E+09,  1.0000000000E+00, -9.9667914135E-02, -1.9502607296E-03,  47,  49],
	[  2.6652519260E+09,  1.0000000000E+00, -9.4953056733E-02, -1.7551493203E-03,  48,  49],
	[  2.8055283432E+09,  1.0000000000E+00, -9.0662612024E-02, -1.5879260732E-03,  48,  50],
	[  2.9458047603E+09,  1.0000000000E+00, -8.6741946358E-02, -1.4435172148E-03,  49,  51],
	[  3.0860811775E+09,  1.0000000000E+00, -8.3145382251E-02, -1.3179527454E-03,  50,  52],
	[  3.2263575947E+09,  1.0000000000E+00, -7.9834442681E-02, -1.2080902830E-03,  51,  53],
	[  3.3666340118E+09,  1.0000000000E+00, -7.6776491069E-02, -1.1114162938E-03,  52,  54],
	[  3.5069104290E+09,  1.0000000000E+00, -7.3943667138E-02, -1.0259008798E-03,  53,  56],
	[  3.6471868461E+09,  1.0000000000E+00, -7.1312046487E-02, -9.4989023927E-04,  55,  57],
	[  3.7874632633E+09,  1.0000000000E+00, -6.8860971047E-02, -8.8202602996E-04,  56,  59],
	[  3.9277396805E+09,  1.0000000000E+00, -6.6572511364E-02, -8.2118420810E-04,  58,  61],
	[  4.0680160976E+09,  1.0000000000E+00, -6.4431031447E-02, -7.6642814617E-04,  60,  65],
	[  4.2082925148E+09,  1.0000000000E+00, -6.2422834143E-02, -7.1697234255E-04,  64, 128],
	[  4.3485689319E+09,  1.0000000000E+00,  0.0000000000E+00,  0.0000000000E+00, 127, 128],
])