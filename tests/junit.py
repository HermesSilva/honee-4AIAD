#!/usr/bin/env python3
from junit_common import *


def create_argparser() -> argparse.ArgumentParser:
    """Creates argument parser to read command line arguments

    Returns:
        argparse.ArgumentParser: Created `ArgumentParser`
    """
    parser = argparse.ArgumentParser('Test runner with JUnit and TAP output')
    parser.add_argument(
        '-c',
        '--ceed-backends',
        type=str,
        nargs='*',
        default=['/cpu/self'],
        help='libCEED backend to use with convergence tests')
    parser.add_argument(
        '-m',
        '--mode',
        type=RunMode,
        action=CaseInsensitiveEnumAction,
        help='Output mode, junit or tap',
        default=RunMode.JUNIT)
    parser.add_argument('-n', '--nproc', type=int, default=1, help='number of MPI processes')
    parser.add_argument('-o', '--output', type=Optional[Path], default=None, help='Output file to write test')
    parser.add_argument('-b', '--junit-batch', type=str, default='', help='Name of JUnit batch for output file')
    parser.add_argument('-np', '--pool-size', type=int, default=1, help='Number of test cases to run in parallel')
    parser.add_argument('-s', '--smartredis_dir', type=str, default='', help='path to SmartSim library, if present')
    parser.add_argument('--has_torch', type=bool, default=False, help='Whether to build with torch')
    parser.add_argument('test', help='Test executable', nargs='?')

    return parser


def diff_csv_comment_function(test_line: str, true_line: str) -> Optional[str]:
    test_line_split = test_line.split(':')[0]
    true_line_split = true_line.split(':')[0]
    diff_output = ''.join(difflib.unified_diff([test_line_split + '\n'],
                                               [true_line_split + '\n'],
                                               tofile='test created file',
                                               fromfile='expected output'))
    return diff_output if diff_output else None


class HoneeSuiteSpec(SuiteSpec):
    def __init__(self, has_torch: bool):
        self.has_torch: bool = has_torch
        self.diff_csv_kwargs: dict = {'comment_func': diff_csv_comment_function, 'rel_tol': 1e-9}

    def get_source_path(self, test: str) -> Path:
        """Compute path to test source file

        Args:
            test (str): Name of test

        Returns:
            Path: Path to source file
        """
        if test.startswith('navierstokes'):
            return (Path('examples') / 'navierstokes').with_suffix('.c')
        else:
            return (Path('tests') / test).with_suffix('.c')

    # get path to executable
    def get_run_path(self, test: str) -> Path:
        """Compute path to built test executable file

        Args:
            test (str): Name of test

        Returns:
            Path: Path to test executable
        """
        return Path('build') / test

    def get_output_path(self, test: str, output_file: str) -> Path:
        """Compute path to expected output file

        Args:
            test (str): Name of test
            output_file (str): File name of output file

        Returns:
            Path: Path to expected output file
        """
        return Path('tests') / 'output' / output_file

    def check_pre_skip(self, test: str, spec: TestSpec, resource: str, nproc: int) -> Optional[str]:
        """Check if a test case should be skipped prior to running, returning the reason for skipping

        Args:
            test (str): Name of test
            spec (TestSpec): Test case specification
            resource (str): libCEED backend
            nproc (int): Number of MPI processes to use when running test case

        Returns:
            Optional[str]: Skip reason, or `None` if test case should not be skipped
        """
        for condition in spec.only:
            if (condition == 'cpu') and ('gpu' in resource):
                return 'CPU only test with GPU backend'
            if condition == 'torch' and not self.has_torch:
                return 'PyTorch only test without USE_TORCH=1'

    def check_post_skip(self, test: str, spec: TestSpec, resource: str, stderr: str) -> Optional[str]:
        """Check if a test case should be allowed to fail, based on its stderr output

        Args:
            test (str): Name of test
            spec (TestSpec): Test case specification
            resource (str): libCEED backend
            stderr (str): Standard error output from test case execution

        Returns:
            Optional[str]: Skip reason, or `None` if unexpeced error
        """
        if 'No SYCL devices of the requested type are available' in stderr:
            return f'SYCL device type not available'
        elif 'You may need to add --download-ctetgen or --download-tetgen' in stderr:
            return f'Tet mesh generator not installed for {test}, {spec.name}'
        return None

    def check_required_failure(self, test: str, spec: TestSpec, resource: str, stderr: str) -> Tuple[str, bool]:
        """Check whether a test case is expected to fail and if it failed expectedly

        Args:
            test (str): Name of test
            spec (TestSpec): Test case specification
            resource (str): libCEED backend
            stderr (str): Standard error output from test case execution

        Returns:
            tuple[str, bool]: Tuple of the expected failure string and whether it was present in `stderr`
        """
        return '', True

    def check_allowed_stdout(self, test: str) -> bool:
        """Check whether a test is allowed to print console output

        Args:
            test (str): Name of test

        Returns:
            bool: True if the test is allowed to print console output
        """
        return False


if __name__ == '__main__':
    args = create_argparser().parse_args()

    # run tests
    if 'smartsim' in args.test:
        has_smartsim: bool = args.smartredis_dir and Path(args.smartredis_dir).is_dir()
        test_cases = []

        if args.mode is RunMode.TAP:
            print(f'1..1')
        if has_smartsim:
            from smartsim_regression_framework import SmartSimTest

            test_framework = SmartSimTest(Path(__file__).parent / 'smartsim_test_dir')
            test_framework.setup()

            is_new_subtest = True
            subtest_ok = True
            for i, backend in enumerate(args.ceed_backends):
                test_cases.append(test_framework.test_junit(backend))
                if is_new_subtest and args.mode == RunMode.TAP:
                    is_new_subtest = False
                    print(f'# Subtest: {test_cases[0].category}')
                    print(f'    1..{len(args.ceed_backends)}')
                print(test_case_output_string(test_cases[i], TestSpec("SmartSim Tests"), args.mode, backend, '', i))
            if args.mode == RunMode.TAP:
                print(f'{"" if subtest_ok else "not "}ok 1 - {test_cases[0].category}')
            test_framework.teardown()
        elif args.mode is RunMode.TAP:
            print(f'ok 1 - # SKIP SmartSim not installed')
        result: TestSuite = TestSuite('SmartSim Tests', test_cases)
    else:
        result: TestSuite = run_tests(
            args.test,
            args.ceed_backends,
            args.mode,
            args.nproc,
            HoneeSuiteSpec(args.has_torch),
            args.pool_size)

    # write output and check for failures
    if args.mode is RunMode.JUNIT:
        write_junit_xml(result, args.output, args.junit_batch)
        if has_failures(result):
            sys.exit(1)
