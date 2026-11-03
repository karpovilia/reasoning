"""Offline recovery and same-prefix checks; never contacts a model or Wikipedia."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from retrieval_control import branch_worker as w

w.configure_imports({})


def reply(content='', calls=None):
    m={'role':'assistant','content':content,'reasoning_content':'preserved reasoning'}
    if calls:
        m['tool_calls']=[{'id':f'c{i}', 'type':'function','function':{'name':'wikipedia_search','arguments':json.dumps({'query':q})}} for i,q in enumerate(calls)]
    return {'choices':[{'message':m,'finish_reason':'stop'}],'usage':{'prompt_tokens':100,'completion_tokens':11,'total_tokens':111}}


class Model:
    def __init__(self, outputs):self.outputs=list(outputs);self.requests=[]
    def call(self, request, request_id):
        self.requests.append((copy.deepcopy(request),request_id))
        value=self.outputs.pop(0)
        if isinstance(value,BaseException):raise value
        return value


class Tools:
    openai_tools=[{'type':'function','function':{'name':'wikipedia_search','parameters':{'type':'object'}}}]
    def __init__(self):self.calls=[];self._path_cache={}
    def execute_tool_call(self,name,args):self.calls.append((name,args));return {'query':json.loads(args)['query'],'results':[]}


class Checks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.out=Path(self.tmp.name)/'out.json'
        self.prefix=[{'role':'system','content':'Answer using tools.'},{'role':'user','content':'Question'},
            {'role':'assistant','content':'','reasoning':'archived reasoning','source':'metadata','tool_calls':[{'id':'old','type':'function','function':{'name':'wikipedia_search','arguments':'{"query":"a"}'}}]},
            {'role':'tool','tool_call_id':'old','name':'wikipedia_search','content':'{"results":[]}'}]
        self.task={'task_id':'task','question_id':'q','benchmark':'hotpotqa','question':'Question','messages':self.prefix,'prefix_sha256':w.digest(self.prefix),'action':'continue','seed':101}
        self.config={'model':'mock','additional_tool_rounds':2,'max_new_tokens':2048,'capture_required':False}
    def tearDown(self):self.tmp.cleanup()

    def test_rounds_count_model_turns_not_parallel_calls(self):
        model=Model([reply(calls=['a','b']),reply(calls=['c','d']),reply('{"concise_answer":"Paris"}')]);tools=Tools()
        result=w.Branch(self.task,self.config,self.out,model,tools).run()
        self.assertEqual(result['additional_tool_rounds'],2);self.assertEqual(len(tools.calls),4)
        self.assertEqual(len(model.requests),3);self.assertEqual(result['usage_totals']['completion_tokens'],33)
        self.assertNotIn('tools',model.requests[-1][0]);self.assertEqual(result['answer'],'Paris')
        w.validate_closed_prefix(result['messages'])

    def test_answer_uses_same_prefix_and_preserves_reasoning(self):
        task={**self.task,'action':'answer'};model=Model([reply('{"concise_answer":"Paris"}')])
        result=w.Branch(task,self.config,self.out,model,Tools()).run();req=model.requests[0][0]
        self.assertEqual(req['messages'][:-1],w.sanitize_messages(self.prefix))
        self.assertEqual(req['messages'][2]['reasoning_content'],'archived reasoning')
        self.assertNotIn('source',req['messages'][2]);self.assertNotIn('tools',req)
        self.assertEqual(result['prefix_sha256'],self.task['prefix_sha256'])

    def test_transport_resume_keeps_accepted_calls(self):
        first=Model([reply(calls=['a']),w.RetryableTransport('outage')]);tools=Tools()
        with self.assertRaises(w.RetryableTransport):w.Branch(self.task,self.config,self.out,first,tools).run()
        self.assertEqual(len(tools.calls),1);self.assertFalse(self.out.exists())
        second=Model([reply('{"concise_answer":"Paris"}')]);result=w.Branch(self.task,self.config,self.out,second,tools).run()
        self.assertEqual(len(tools.calls),1);self.assertEqual(len(second.requests),1)
        self.assertEqual(result['usage_totals']['completion_tokens'],22)
        self.assertEqual(result['unconfirmed_llm_attempts'],1)
        self.assertEqual(second.requests[0][0]['seed'],102)

    def test_interruption_after_response_commit_replays_without_llm(self):
        first=Model([reply(calls=['a'])]);tools=Tools();branch=w.Branch(self.task,self.config,self.out,first,tools)
        original=branch.journal.perform
        def interrupted(key,request,operation):
            result=original(key,request,operation)
            if key=='llm_0000':raise KeyboardInterrupt()
            return result
        branch.journal.perform=interrupted
        with self.assertRaises(KeyboardInterrupt):branch.run()
        attempt=branch.journal.path/'llm_0000.attempt0000.json'
        interrupted_receipt=w.load(attempt)
        interrupted_receipt['status']='started'
        interrupted_receipt.pop('wall_s',None)
        interrupted_receipt.pop('finished_at',None)
        w.atomic_json(attempt,interrupted_receipt)
        second=Model([reply('{"concise_answer":"Paris"}')]);result=w.Branch(self.task,self.config,self.out,second,tools).run()
        self.assertEqual(len(second.requests),1);self.assertEqual(len(tools.calls),1)
        self.assertEqual(result['usage_totals']['completion_tokens'],22)
        self.assertEqual(w.load(attempt)['status'],'accepted_recovered')
        self.assertGreater(result['active_call_wall_s'],0)

    def test_repair_bound_and_idempotent_finished_output(self):
        task={**self.task,'action':'answer'};model=Model([reply('wrong')]*3)
        result=w.Branch(task,self.config,self.out,model,Tools()).run()
        self.assertEqual(len(model.requests),3);self.assertEqual(result['status'],'semantic_failure')
        self.assertEqual(result['answer'],'')
        again=Model([]);self.assertEqual(w.Branch(task,self.config,self.out,again,Tools()).run(),result)
        self.assertEqual(again.requests,[])

    def test_pending_prefix_gold_and_changed_config_rejected(self):
        bad={**self.task,'messages':self.prefix[:-1]};bad['prefix_sha256']=w.digest(bad['messages'])
        with self.assertRaises(w.ConfigurationError):w.Branch(bad,self.config,self.out,Model([]),Tools())
        with self.assertRaises(w.ConfigurationError):w.Branch({**self.task,'gold_answer':'X'},self.config,self.out,Model([]),Tools())
        w.Branch(self.task,self.config,self.out,Model([]),Tools())
        with self.assertRaises(w.ConfigurationError):w.Branch(self.task,{**self.config,'max_new_tokens':8},self.out,Model([]),Tools())

    def test_http503_is_transport_and_http400_is_configuration(self):
        import httpx
        for status,kind in [(503,w.RetryableTransport),(400,w.ConfigurationError)]:
            client=w.HTTPModel({'base_url':'http://localhost:1/v1'})
            client.client=httpx.Client(transport=httpx.MockTransport(lambda req:httpx.Response(status,json={'error':'fixture'})))
            with self.assertRaises(kind):client.call({'model':'mock'},'fixture')

    def test_tool_cache_is_immutable_and_recoverable(self):
        import httpx
        count=[]
        def respond(req):
            count.append(str(req.url))
            return httpx.Response(200,json={'query':{'search':[{'title':'Paris','snippet':'France'}]}},headers={'etag':'fixed-version'})
        client=httpx.Client(base_url='http://wiki.fixture',transport=httpx.MockTransport(respond))
        cfg={'tool_base_url':'http://wiki.fixture','wiki_backend':'wikipedia','tool_http_cache_dir':str(Path(self.tmp.name)/'cache')}
        with patch('httpx.Client',return_value=client):tools=w.make_wikipedia(cfg)
        a=tools.execute_tool_call('wikipedia_search','{"query":"capital"}')
        b=tools.execute_tool_call('wikipedia_search','{"query":"capital"}')
        self.assertEqual(a,b);self.assertEqual(len(count),1)
        self.assertEqual(tools.client.history[0]['body_sha256'],tools.client.history[1]['body_sha256'])
        (Path(cfg['tool_http_cache_dir'])/'FROZEN.json').write_text('{}')
        self.assertEqual(tools.execute_tool_call('wikipedia_search','{"query":"capital"}'),a)
        with self.assertRaises(w.ConfigurationError):tools.execute_tool_call('wikipedia_search','{"query":"other"}')

    def test_tool_http503_is_not_silenced_as_empty_search(self):
        import httpx
        client=httpx.Client(base_url='http://wiki.fixture',transport=httpx.MockTransport(lambda req:httpx.Response(503)))
        with patch('httpx.Client',return_value=client):tools=w.make_wikipedia({'tool_base_url':'http://wiki.fixture','wiki_backend':'kiwix'})
        with self.assertRaises(w.RetryableTransport):tools.execute_tool_call('wikipedia_search','{"query":"capital"}')

    def test_required_capture_failure_stops_after_durable_response(self):
        model=Model([reply('{"concise_answer":"Paris"}')]);cfg={**self.config,'capture_required':True}
        with self.assertRaises(w.ConfigurationError):w.Branch({**self.task,'action':'answer'},cfg,self.out,model,Tools()).run()
        self.assertEqual(len(model.requests),1)
        self.assertTrue((Path(str(self.out)+'.journal')/'llm_0000.complete.json').exists())
        again=Model([])
        with self.assertRaises(w.ConfigurationError):w.Branch({**self.task,'action':'answer'},cfg,self.out,again,Tools()).run()
        self.assertEqual(len(again.requests),0)


if __name__=='__main__':unittest.main(verbosity=2)
