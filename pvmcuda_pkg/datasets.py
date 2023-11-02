# GPU PVM Implementation
# (C) 2023 Filip Piekniewski 
# filip@piekniewski.info

sets = {
    "face_training": ["face01.pkl.zip",
                      "face03.pkl.zip",
                      "no_target.pkl.zip",
                      "face16.pkl.zip",
                      "face17.pkl.zip",
                      "face18.pkl.zip",
                      "face19.pkl.zip",
                      "no_target_01.pkl.zip",
                      "face20.pkl.zip",
                      "face21.pkl.zip",
                      ],
    "face_testing": ["face02.pkl.zip",
                     "face04.pkl.zip",
                     "face05.pkl.zip",
                     "face06.pkl.zip",
                     "face07.pkl.zip",
                     "face08.pkl.zip",
                     "face09.pkl.zip",
                     "face10.pkl.zip",
                     "face11.pkl.zip",
                     "face12.pkl.zip",
                     "face13.pkl.zip",
                     "face14.pkl.zip",
                     "face15.pkl.zip",
                     "face22.pkl.zip",
                     "face23.pkl.zip",
                     "face24.pkl.zip",
                     ],
    "face_ex_testing": ["face25.pkl.zip",
                        "face26.pkl.zip",
                        "face27.pkl.zip",
                        "face28.pkl.zip",
                        "face29.pkl.zip",
                        "face30.pkl.zip",
                        "face31.pkl.zip",
                        "face32.pkl.zip",
                        "face33.pkl.zip",
                        ],
    "face_additional": [],
    "green_ball_training": ["green_ball_long.pkl.zip",
                            "green_ball_on_grass.pkl.zip",
                            "green_ball_test_14.pkl.zip",
                            "green_ball_test_15.pkl.zip",
                            "green_ball_test_16.pkl.zip",
                            ],
    "green_ball_testing": ["green_ball_test.pkl.zip",
                           "green_ball_test_01.pkl.zip",
                           "green_ball_test_02.pkl.zip",
                           "green_ball_test_03.pkl.zip",
                           "green_ball_test_04.pkl.zip",
                           "green_ball_test_05.pkl.zip",
                           "green_ball_test_06.pkl.zip",
                           "green_ball_test_07.pkl.zip",
                           "green_ball_test_08.pkl.zip",
                           "green_ball_test_09.pkl.zip",
                           "green_ball_test_10.pkl.zip",
                           "green_ball_test_11.pkl.zip",
                           "green_ball_test_12.pkl.zip",
                           "green_ball_test_13.pkl.zip",
                           "green_ball_01_small.pkl.zip",
                           "green_ball_bc_office.pkl.zip",
                           ],
    "green_ball_ex_testing": ["green_ball_test_17.pkl.zip",
                              "green_ball_test_18.pkl.zip",
                              "green_ball_test_19.pkl.zip",
                              "green_ball_test_20.pkl.zip",
                              "green_ball_test_21.pkl.zip",
                              "green_ball_test_22.pkl.zip",
                              "green_ball_test_23.pkl.zip",
                              "green_ball_test_24.pkl.zip",
                              "green_ball_test_25.pkl.zip",
                              "green_ball_test_26.pkl.zip",
                              "green_ball_test_27.pkl.zip",
                              "green_ball_test_28.pkl.zip",
                              "green_ball_test_29.pkl.zip",
                              ],
    "green_ball_additional": ["blue_ball_on_grass_daytime.pkl.zip",
                              "blue_ball_at_home_02.pkl.zip",
                              "blue_ball_at_home_01.pkl.zip",
                              ],
    "stop_sign_training": ["stop01.pkl.zip",
                           "stop03.pkl.zip",
                           "stop05.pkl.zip",
                           "stop07.pkl.zip",
                           "stop09.pkl.zip",
                           "stop11.pkl.zip",
                           "stop13.pkl.zip",
                           "stop15.pkl.zip",
                           "stop17.pkl.zip",
                           "stop19.pkl.zip",
                           "stop21.pkl.zip",
                           "stop23.pkl.zip",
                           "stop25.pkl.zip",
                           "stop27.pkl.zip",
                           "stop29.pkl.zip",
                           "stop32.pkl.zip",
                           "stop34.pkl.zip",
                           "stop36.pkl.zip",
                           "stop38.pkl.zip",
                           "stop40.pkl.zip",
                           ],
    "stop_sign_testing": ["stop02.pkl.zip",
                          "stop04.pkl.zip",
                          "stop06.pkl.zip",
                          "stop08.pkl.zip",
                          "stop10.pkl.zip",
                          "stop12.pkl.zip",
                          "stop14.pkl.zip",
                          "stop16.pkl.zip",
                          "stop18.pkl.zip",
                          "stop20.pkl.zip",
                          "stop22.pkl.zip",
                          "stop24.pkl.zip",
                          "stop26.pkl.zip",
                          "stop28.pkl.zip",
                          "stop30.pkl.zip",
                          "stop33.pkl.zip",
                          "stop35.pkl.zip",
                          "stop37.pkl.zip",
                          "stop39.pkl.zip",
                          ],
    "stop_sign_additional": [],
    "stop_sign_ex_testing": ["stop41.pkl.zip",
                             "stop42.pkl.zip",
                             "stop43.pkl.zip",
                             "stop44.pkl.zip",
                             "stop45.pkl.zip",
                             "stop46.pkl.zip",
                             "stop47.pkl.zip",
                             "stop48.pkl.zip",
                             "stop49.pkl.zip",
                             "stop50.pkl.zip",
                             "stop51.pkl.zip",
                             ],
    "short_training": ["stop40.pkl.zip"],
    "short_testing": ["stop41.pkl.zip"],
    "short_additional": [],

    "non_spec_training": ["no_target.pkl.zip",
                          "no_target_01.pkl.zip",
                          "green_ball_long.pkl.zip",
                          "stop01.pkl.zip",
                          "green_ball_on_grass.pkl.zip",
                          "stop03.pkl.zip",
                          "face01.pkl.zip",
                          "stop05.pkl.zip",
                          "face03.pkl.zip",
                          "stop07.pkl.zip",
                          "face16.pkl.zip",
                          "stop09.pkl.zip",
                          "face17.pkl.zip",
                          "stop11.pkl.zip",
                          "face18.pkl.zip",
                          "stop13.pkl.zip",
                          "green_ball_test_14.pkl.zip",
                          "stop15.pkl.zip",
                          "green_ball_test_15.pkl.zip",
                          "stop17.pkl.zip",
                          "green_ball_test_16.pkl.zip",
                          "stop19.pkl.zip",
                          "stop21.pkl.zip",
                          "stop23.pkl.zip",
                          ],
    "non_spec_testing": ["stop41.pkl.zip"],
    "non_spec_additional": [],
    "camvid_full": [
        "0001TP.zip",
        "0006R0.zip",
        "0016E5.zip",
        "Seq05VD.zip"
    ],
    "camvid_small": [
        "0001TP_small.zip"
    ],
    "camvid_1st": [
        "0001TP.zip"
    ],
    "camvid_05x_training": [
        "0001TP_05x_training.zip",
        "0006R0_05x_training.zip",
        "0016E5_05x_training.zip",
        "Seq05VD_05x_training.zip"
    ],
    "camvid_05x_testing": [
        "0001TP_05x_testing.zip",
        "0006R0_05x_testing.zip",
        "0016E5_05x_testing.zip",
        "Seq05VD_05x_testing.zip"
    ],
    "carla_training": [
        "calra_001_000.zip",
        "calra_001_001.zip",
        "calra_001_002.zip",
        "calra_001_003.zip",
        "calra_001_004.zip",
        "calra_001_005.zip",
        "calra_001_006.zip",
        "calra_001_007.zip",
        "calra_001_008.zip",
        "calra_001_009.zip",
        "calra_001_010.zip",
        "calra_001_011.zip",
        "calra_001_012.zip",
        "calra_001_013.zip",
        "calra_001_014.zip",
        "calra_001_015.zip",
        "calra_001_016.zip",
        "calra_001_017.zip",
        "calra_001_018.zip",
        "calra_001_019.zip",
        "calra_001_020.zip",
        "calra_001_021.zip",
        "calra_001_022.zip",
        "calra_001_023.zip",
        "calra_001_024.zip",
        "calra_001_025.zip",
        "calra_001_026.zip",
        "calra_001_027.zip",
        "calra_001_028.zip",
        "calra_001_029.zip",
        "calra_001_030.zip",
        "calra_001_031.zip",
        "calra_001_032.zip",
        "calra_001_033.zip",
        "calra_001_034.zip",
        "calra_001_035.zip",
        "calra_001_036.zip",
        "calra_001_037.zip",
        "calra_001_038.zip",
        "calra_001_039.zip",
        "calra_001_040.zip",
        "calra_001_041.zip",
        "calra_001_042.zip",
        "calra_001_043.zip",
        "calra_001_044.zip",
        "calra_001_045.zip",
    ],
    "carla_testing":[
        "calra_006_000_test.zip",
        "calra_006_001_test.zip",
        "calra_006_002_test.zip",
        "calra_006_003_test.zip",
        "calra_006_004_test.zip",
        "calra_006_005_test.zip",
        "calra_006_006_test.zip",
        "calra_006_007_test.zip",
        "calra_006_008_test.zip",
    ],
    "carla_testing_single": [
        "calra_006_008_test.zip",
    ],
    "carla_testing_single0": [
        "calra_006_006_test.zip",
    ]


}


sets["stop_sign_full_testing"] = sets["stop_sign_testing"] + sets["stop_sign_ex_testing"]
sets["green_ball_full_testing"] = sets["green_ball_testing"] + sets["green_ball_ex_testing"]
sets["face_full_testing"] = sets["face_testing"] + sets["face_ex_testing"]

sets["stop_sign_fast_testing"] = [sets["stop_sign_testing"][0]]
sets["green_ball_fast_testing"] = [sets["green_ball_testing"][0]]
sets["face_fast_testing"] = [sets["face_testing"][0]]
