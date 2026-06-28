module ai_numeric
    use, intrinsic :: iso_c_binding, only: c_double, c_int64_t
    implicit none
contains

    subroutine compute_prediction_metrics(y_true, y_pred, n, mae, rmse, max_error) &
            bind(C, name="compute_prediction_metrics")
        integer(c_int64_t), value, intent(in) :: n
        real(c_double), intent(in) :: y_true(*)
        real(c_double), intent(in) :: y_pred(*)
        real(c_double), intent(out) :: mae
        real(c_double), intent(out) :: rmse
        real(c_double), intent(out) :: max_error

        integer(c_int64_t) :: i
        real(c_double) :: error_value
        real(c_double) :: absolute_error
        real(c_double) :: absolute_sum
        real(c_double) :: squared_sum

        mae = 0.0_c_double
        rmse = 0.0_c_double
        max_error = 0.0_c_double
        if (n <= 0_c_int64_t) return

        absolute_sum = 0.0_c_double
        squared_sum = 0.0_c_double
        do i = 1_c_int64_t, n
            error_value = y_pred(i) - y_true(i)
            absolute_error = abs(error_value)
            absolute_sum = absolute_sum + absolute_error
            squared_sum = squared_sum + error_value * error_value
            if (absolute_error > max_error) max_error = absolute_error
        end do

        mae = absolute_sum / real(n, c_double)
        rmse = sqrt(squared_sum / real(n, c_double))
    end subroutine compute_prediction_metrics


    subroutine average_waveform_pairs(time_values, voltage_values, n_runs, n_points, &
            time_average, voltage_average) bind(C, name="average_waveform_pairs")
        integer(c_int64_t), value, intent(in) :: n_runs
        integer(c_int64_t), value, intent(in) :: n_points
        real(c_double), intent(in) :: time_values(*)
        real(c_double), intent(in) :: voltage_values(*)
        real(c_double), intent(out) :: time_average(*)
        real(c_double), intent(out) :: voltage_average(*)

        integer(c_int64_t) :: run_index
        integer(c_int64_t) :: point_index
        integer(c_int64_t) :: flat_index
        real(c_double) :: time_sum
        real(c_double) :: voltage_sum

        if (n_runs <= 0_c_int64_t .or. n_points <= 0_c_int64_t) return

        do point_index = 1_c_int64_t, n_points
            time_sum = 0.0_c_double
            voltage_sum = 0.0_c_double
            do run_index = 1_c_int64_t, n_runs
                ! Python passes a C-contiguous [run, point] matrix.
                flat_index = (run_index - 1_c_int64_t) * n_points + point_index
                time_sum = time_sum + time_values(flat_index)
                voltage_sum = voltage_sum + voltage_values(flat_index)
            end do
            time_average(point_index) = time_sum / real(n_runs, c_double)
            voltage_average(point_index) = voltage_sum / real(n_runs, c_double)
        end do
    end subroutine average_waveform_pairs

end module ai_numeric
