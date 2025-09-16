from playwright.async_api import async_playwright
import os
import asyncio
from datetime import datetime

async def run_playwright_test():
    async with async_playwright() as p:
        # Define paths and setup
        job_file_path = r"C:\Users\user\Downloads\TUGAS AKHIR\archive (16)\job_data.csv"
        resume_files = [
            r"C:\Users\user\Downloads\TUGAS AKHIR\archive (16)\Resume_1.pdf",
            r"C:\Users\user\Downloads\TUGAS AKHIR\archive (16)\Resume_2.pdf",
            r"C:\Users\user\Downloads\TUGAS AKHIR\archive (16)\Resume_3.pdf",
            r"C:\Users\user\Downloads\TUGAS AKHIR\archive (16)\Resume_4.pdf",
            r"C:\Users\user\Downloads\TUGAS AKHIR\archive (16)\Resume_5.pdf"
        ]
        video_dir = 'videos'  # Directory to save videos
        report_file_path = 'test_report.txt'  # Path for the text report

        # Ensure the video directory exists
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)

        # Launch the browser with video recording enabled
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(record_video_dir=video_dir)
        page = await context.new_page()

        # Function to log messages with timestamps to the report file
        def log_to_report(message):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(report_file_path, 'a') as report_file:
                report_file.write(f"{timestamp} - {message}\n")

        log_to_report("Test started.")

        # Navigate to the Streamlit app URL
        start_time = datetime.now()
        await page.goto('http://localhost:8502')  # Replace with your Streamlit app URL
        await page.wait_for_load_state('networkidle')
        load_time = datetime.now() - start_time
        log_to_report(f"Page loaded in {load_time}.")

        # Check file paths
        def check_files():
            log_to_report("Checking file paths:")
            log_to_report(f"Job file exists: {os.path.exists(job_file_path)}")
            for resume_file_path in resume_files:
                log_to_report(f"Resume file {resume_file_path} exists: {os.path.exists(resume_file_path)}")
        
        check_files()

        # Upload Job Data CSV
        start_time = datetime.now()
        file_upload_trigger = await page.query_selector('input[type="file"][accept=".csv"]')
        if file_upload_trigger:
            await file_upload_trigger.set_input_files(job_file_path)
            log_to_report("Job data CSV uploaded successfully.")
        else:
            log_to_report("File input for job data CSV not found.")
        upload_job_time = datetime.now() - start_time
        log_to_report(f"Job data CSV upload time: {upload_job_time}.")

        # Upload Resumes
        start_time = datetime.now()
        file_upload_trigger = await page.query_selector('input[type="file"][accept=".pdf"]')
        if file_upload_trigger:
            for resume_file_path in resume_files:
                if os.path.exists(resume_file_path):
                    await file_upload_trigger.set_input_files(resume_file_path)
                    log_to_report(f"Resume file {resume_file_path} uploaded successfully.")
                else:
                    log_to_report(f"Resume PDF file not found at {resume_file_path}.")
        else:
            log_to_report("File input for resumes not found.")
        upload_resume_time = datetime.now() - start_time
        log_to_report(f"Resume files upload time: {upload_resume_time}.")

        # Take a screenshot
        start_time = datetime.now()
        await page.screenshot(path='screenshot_before_wait.png')
        screenshot_time = datetime.now() - start_time
        log_to_report(f"Screenshot taken in {screenshot_time}.")

        # Set the value of k
        start_time = datetime.now()
        k_value_input = await page.wait_for_selector('input[data-testid="stNumberInput-Input"]', timeout=90000)
        if k_value_input:
            await k_value_input.fill("1")  # Set the desired value of k
            log_to_report("Value of k set to 1.")
        else:
            log_to_report("Number input field for k value not found.")
        set_k_time = datetime.now() - start_time
        log_to_report(f"Setting k value time: {set_k_time}.")

        # Use step-up button if necessary
        start_time = datetime.now()
        step_up_button = await page.query_selector('button[data-testid="stNumberInput-StepUp"]')
        if step_up_button:
            await step_up_button.click()  # Click the button to increase the value
            log_to_report("Step-up button clicked to increase k value.")
        else:
            log_to_report("Step-up button not found.")
        step_up_time = datetime.now() - start_time
        log_to_report(f"Step-up button time: {step_up_time}.")

        # Set the number of companies
        start_time = datetime.now()
        num_companies_input = await page.wait_for_selector('input[placeholder="Enter the number of companies to randomly select"]', timeout=90000)
        if num_companies_input:
            await num_companies_input.fill("3")  # Set the desired number of companies
            log_to_report("Number of companies set to 3.")
        else:
            log_to_report("Input field for number of companies not found.")
        set_num_companies_time = datetime.now() - start_time
        log_to_report(f"Setting number of companies time: {set_num_companies_time}.")

        # Click the button to start the matching process
        start_time = datetime.now()
        match_button = await page.wait_for_selector('button:has-text("Match Resumes to Jobs")', timeout=90000)
        if match_button:
            await match_button.click()
            log_to_report("Match button clicked to start the matching process.")
        else:
            log_to_report("Match button not found.")
        match_process_time = datetime.now() - start_time
        log_to_report(f"Matching process time: {match_process_time}.")

        # Wait for results to process
        await asyncio.sleep(10)  # Adjust the sleep time based on your application's processing time

        log_to_report("Test completed.")

        # Close the browser
        await browser.close()

if __name__ == "__main__":
    asyncio.run(run_playwright_test())
