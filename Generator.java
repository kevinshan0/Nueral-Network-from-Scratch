import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class Generator {

    private static final String FILE_NAME = "training_data.csv";
    private static final int GRID_SIZE = 9;
    private static final int TRAINING_DATA_SIZE = 10000;

    // Method to write data to CSV file
    private static void writeToCSV(List<int[][]> data) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(FILE_NAME, true))) {
            writer.write(Arrays.deepToString(data.get(0)) + "-" + Arrays.deepToString(data.get(1)));
            writer.newLine();

            System.out.println("Data written successfully to " + FILE_NAME);
        } catch (IOException e) {
            System.err.println("Error writing to file: " + e.getMessage());
        }
    }

    // Method to fill the grid using backtracking
    private static boolean fillGrid(int[][] grid) {
        for (int row = 0; row < GRID_SIZE; row++) {
            for (int col = 0; col < GRID_SIZE; col++) {
                if (grid[row][col] == 0) {
                    // Randomize numbers for variation
                    int[] numbers = getShuffledNumbers();
                    for (int num : numbers) {
                        if (isValid(grid, row, col, num)) {
                            grid[row][col] = num;
                            if (fillGrid(grid)) {
                                return true;
                            }
                            grid[row][col] = 0; // Backtrack
                        }
                    }
                    return false; // No valid number found, backtrack
                }
            }
        }
        return true; // All cells filled
    }

    // Check if placing a number is valid
    private static boolean isValid(int[][] grid, int row, int col, int num) {
        // Check row
        for (int x = 0; x < GRID_SIZE; x++) {
            if (grid[row][x] == num) return false;
        }

        // Check column
        for (int x = 0; x < GRID_SIZE; x++) {
            if (grid[x][col] == num) return false;
        }

        // Check 3x3 subgrid
        int startRow = row - row % 3;
        int startCol = col - col % 3;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (grid[startRow + i][startCol + j] == num) return false;
            }
        }

        return true;
    }

    // Generate shuffled numbers 1-9
    private static int[] getShuffledNumbers() {
        int[] numbers = new int[GRID_SIZE];
        for (int i = 0; i < GRID_SIZE; i++) {
            numbers[i] = i + 1;
        }

        // Shuffle the array
        Random random = new Random();
        for (int i = numbers.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int temp = numbers[i];
            numbers[i] = numbers[j];
            numbers[j] = temp;
        }

        return numbers;
    }

    // generate half-filled grid
    private static int[][] removeHalfValues(int[][] grid) {
        Random random = new Random();

        for (int row = 0; row < GRID_SIZE; row++) {
            for (int col = 0; col < GRID_SIZE; col++) {
                if (random.nextDouble() < 0.5) {
                    grid[row][col] = 0;
                }
            }
        }

        return grid;
    }

    public static void main(String[] args) {
        for (int index = 0; index < TRAINING_DATA_SIZE; index++) {
            int[][] grid = new int[GRID_SIZE][GRID_SIZE];

            if (fillGrid(grid)) {
                List<int[][]> half_and_full_grids = new ArrayList<>();

                int[][] half_filled_grid = new int[GRID_SIZE][GRID_SIZE];
                for (int i = 0; i < GRID_SIZE; i++) {
                    half_filled_grid[i] = Arrays.copyOf(grid[i], grid[i].length);
                }

                half_and_full_grids.add(removeHalfValues(half_filled_grid));
                half_and_full_grids.add(grid);

                writeToCSV(half_and_full_grids);
            }
        }
    }
}