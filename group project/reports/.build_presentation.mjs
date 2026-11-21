import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const project = path.resolve('C:/Users/ROG/Desktop/课程/Y3sem1/Big Data Analytics/DSAI4205/group project');
const skill = 'C:/Users/ROG/.codex/plugins/cache/openai-primary-runtime/presentations/26.1112.11217/skills/presentations';
const dependencies = 'C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1112-737-0_26.1112.737.0_x64__3k8sg7r9htsxt/dependencies';
const {Presentation, PresentationFile} = await import(pathToFileURL(path.join(dependencies, 'node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const {finalizePresentation} = await import(pathToFileURL(path.join(skill, 'container_tools/artifact_tool_utils.mjs')).href);
const content = JSON.parse(await fs.readFile(path.join(project, 'reports/.presentation_draft_content.json'), 'utf8'));
const staging = path.join(project, 'reports/.presentation_build');
await fs.mkdir(staging, {recursive:true});
await fs.mkdir(path.join(project, 'deliverables'), {recursive:true});
const pres = Presentation.create({slideSize:{width:1280,height:720}});
for (let i=0; i<content.length; i++) {
  const s=pres.slides.add(); s.background.fill='#F9FAFC';
  for (const a of content[i].shapes) {
    const sh=s.shapes.add({geometry:'textbox',position:{left:a.x,top:a.y,width:a.w,height:a.h},fill:'none',line:{fill:'none',width:0}});
    sh.text=a.text;
    const accent=(a.size>=36 && a.y>170) || (a.size>=29 && a.bold && a.y>170);
    sh.text.style={fontSize:a.size||24,bold:!!a.bold,typeface:'Arial',color:accent?'#175178':a.size<=21?'#56687B':'#10233F',verticalAlignment:'middle'};
  }
  if(i===2 || i===4) {
    const filename=i===2?'01_class_distribution.png':'04_surface_features.png';
    const bytes=await fs.readFile(path.join(project,'figures',filename));
    s.images.add({blob:bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),contentType:'image/png',alt:filename,
      position:i===2?{left:65,top:180,width:820,height:445}:{left:55,top:175,width:800,height:455},fit:'contain'});
  }
  s.speakerNotes.text=content[i].notes;
  const png=await pres.export({slide:s,format:'png',scale:1});
  await fs.writeFile(path.join(staging,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
}
const draft=path.join(staging,'candidate.pptx');
await (await PresentationFile.exportPptx(pres)).save(draft);
const result=await finalizePresentation({workspaceDir:project,candidatePath:draft,
  finalPath:path.join(project,'deliverables/presentation_working_draft.pptx'),
  pythonExecutable:path.join(dependencies,'python/python.exe'),
  integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
  fontPolicy:{basis:'design',families:['Arial']},verifyArtifactToolImport:true,
  receiptPath:path.join(staging,'validation.json')});
console.log(JSON.stringify(result));
