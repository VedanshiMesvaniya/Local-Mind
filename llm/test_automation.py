# test_automation.py
import time
import re
import json
import csv
import httpx
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import asyncio

class LocalMindBenchmark:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.results = []
        
    def parse_question_file(self, file_path: str) -> List[Dict]:
        """Parse question file and extract questions and expected answers (for reference)."""
        questions = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by delimiter (====)
        blocks = re.split(r'=+\n', content)
        
        for block in blocks:
            block = block.strip()
            if not block:
                continue
                
            # Extract components
            question_match = re.search(r'Question:\s*(.+?)(?=Expected Answer:|$)', block, re.DOTALL)
            expected_match = re.search(r'Expected Answer:\s*(.+?)(?=Evaluation Criteria:|$)', block, re.DOTALL)
            
            if question_match:
                questions.append({
                    'question': question_match.group(1).strip(),
                    'expected_answer': expected_match.group(1).strip() if expected_match else "",
                })
        
        return questions
    
    async def query_localmind(self, question: str) -> Dict:
        """Query LocalMind API via streaming to measure exact TTFT and Total Time."""
        payload = {
            "prompt": question,
            "chat_history": [],
            "user_id": "benchmark_user"
        }
        
        start_time = time.time()
        ttft = None
        full_response = ""
        error_msg = None
        
        try:
            # 300s timeout just in case, though your 10-core should be fast
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream("POST", f"{self.api_url}/query", json=payload) as response:
                    if response.status_code == 200:
                        # Stream token-by-token to capture the exact moment generation starts
                        async for chunk in response.aiter_text():
                            if chunk:
                                if ttft is None:
                                    ttft = time.time() - start_time
                                full_response += chunk
                    else:
                        error_msg = f"API Error: Status {response.status_code}"
                        full_response = error_msg
                        
        except Exception as e:
            error_msg = f"Connection Error: {str(e)}"
            full_response = error_msg
            
        total_time = time.time() - start_time
        
        return {
            "actual_answer": full_response,
            "ttft_seconds": round(ttft, 2) if ttft else None,
            "total_time_seconds": round(total_time, 2),
            "error": error_msg
        }
    
    async def run_test_suite(self, question_files: List[str]) -> Dict:
        """Run complete test suite and collect raw metrics."""
        all_questions = []
        
        for file_path in question_files:
            questions = self.parse_question_file(file_path)
            for q in questions:
                q['source_file'] = Path(file_path).name
            all_questions.extend(questions)
        
        print(f"\n{'='*80}")
        print(f"LocalMind Benchmarking Suite (No Internal Scoring)")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Questions: {len(all_questions)}")
        print(f"{'='*80}\n")
        
        total_time = 0
        total_ttft = 0
        ttft_count = 0
        
        for idx, test_case in enumerate(all_questions, 1):
            print(f"\n[Test {idx}/{len(all_questions)}]")
            print(f"Question: {test_case['question'][:80]}...")
            
            # Query the system
            result = await self.query_localmind(test_case['question'])
            
            total_time += result['total_time_seconds']
            if result['ttft_seconds'] is not None:
                total_ttft += result['ttft_seconds']
                ttft_count += 1
            
            # Store result
            record = {
                'test_number': idx,
                'source_file': test_case['source_file'],
                'question': test_case['question'],
                'expected_answer': test_case['expected_answer'],
                'actual_answer': result['actual_answer'],
                'ttft_seconds': result['ttft_seconds'],
                'total_time_seconds': result['total_time_seconds'],
                'error': result['error']
            }
            self.results.append(record)
            
            # Print immediate feedback
            print(f"TTFT: {result['ttft_seconds']}s | Total Time: {result['total_time_seconds']}s")
            if result['error']:
                print(f"❌ ERROR: {result['error']}")
            else:
                print(f"✅ Completed. Response length: {len(result['actual_answer'])} chars")
        
        # Generate summary
        avg_total = total_time / len(all_questions) if all_questions else 0
        avg_ttft = total_ttft / ttft_count if ttft_count > 0 else 0
        
        summary = {
            'total_tests': len(all_questions),
            'successful_tests': ttft_count,
            'failed_tests': len(all_questions) - ttft_count,
            'total_time_seconds': round(total_time, 2),
            'average_total_time': round(avg_total, 2),
            'average_ttft': round(avg_ttft, 2),
            'timestamp': datetime.now().isoformat()
        }
        
        return summary
    
    def generate_report(self, summary: Dict, output_dir: str = "."):
        """Generate detailed JSON and CSV reports for external comparison."""
        report = {
            'summary': summary,
            'detailed_results': self.results
        }
        
        json_path = Path(output_dir) / "benchmark_report.json"
        csv_path = Path(output_dir) / "benchmark_report.csv"
        
        # 1. Save Full JSON (Contains complete, untruncated answers)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        # 2. Save Flattened CSV (Truncated for easy viewing in Excel/Sheets)
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Test #', 'Source File', 'Question', 'Expected Answer (Reference)', 
                'Actual Answer', 'TTFT (s)', 'Total Time (s)', 'Error'
            ])
            for r in self.results:
                # Truncate long text so the CSV remains readable in spreadsheet apps
                exp_trunc = r['expected_answer'][:300] + ('...' if len(r['expected_answer']) > 300 else '')
                act_trunc = r['actual_answer'][:1000] + ('...' if len(r['actual_answer']) > 1000 else '')
                
                writer.writerow([
                    r['test_number'],
                    r['source_file'],
                    r['question'],
                    exp_trunc,
                    act_trunc,
                    r['ttft_seconds'],
                    r['total_time_seconds'],
                    r['error'] or ''
                ])
        
        # Print summary
        print(f"\n{'='*80}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*80}")
        print(f"Total Tests:        {summary['total_tests']}")
        print(f"Successful:         {summary['successful_tests']}")
        print(f"Failed/Errors:      {summary['failed_tests']}")
        print(f"Total Time:         {summary['total_time_seconds']}s")
        print(f"Avg Total Time:     {summary['average_total_time']}s")
        print(f"Avg TTFT:           {summary['average_ttft']}s")
        print(f"{'='*80}")
        print(f"\nFull JSON report saved to: {json_path}")
        print(f"Readable CSV report saved to: {csv_path}")
        
        return report

async def main():
    tester = LocalMindBenchmark(api_url="http://localhost:8000")
    
    test_files = [
        "questions_20.txt",
        "questions_40.txt"
    ]
    
    for file in test_files:
        if not Path(file).exists():
            print(f"Error: {file} not found!")
            return
    
    summary = await tester.run_test_suite(test_files)
    tester.generate_report(summary)

if __name__ == "__main__":
    asyncio.run(main())