// tests/test_main.cpp
// Main entry point for Google Test unit tests

#include <gtest/gtest.h>

// Google Test will automatically discover and run all tests
// You don't need to manually register them

// This is the main function that runs all tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// Example test to verify setup
TEST(SetupTest, BasicAssertion) {
    EXPECT_EQ(1 + 1, 2);
    EXPECT_TRUE(true);
}

TEST(SetupTest, StringComparison) {
    std::string expected = "QuinnCluster";
    std::string actual = "QuinnCluster";
    EXPECT_EQ(expected, actual);
}